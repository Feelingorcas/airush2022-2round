import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from BERTclassifier import essemble_model
from MPL_pytorch.data import DATASET_GETTERS
import util
import dataset
import BERT

from MPL_pytorch.models import WideResNet, ModelEMA
from MPL_pytorch.utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict)

from nsml import DATASET_PATH
import nsml

from sklearn.model_selection import train_test_split

################## dataset ###########################
train_dataset_path = DATASET_PATH + '/train'
all_x = np.load(train_dataset_path + '/train_data/all_x.npy').astype(np.int64)  # Pre-training data
train_x = np.load(train_dataset_path + '/train_data/x.npy').astype(np.int64)
train_y = np.load(train_dataset_path + '/train_label').astype(np.int64)

VOCAB_SIZE = 5003
CLASS_NUM = 10
PAD = 0
MASK = 5001
CLR = 5002
####################### nsml ##################################

def bind_model(model, device):
    def save(dir_name):
        torch.save(model.state_dict(), dir_name + f'/params.pkl')

    def load(dir_name):
        params_fname = dir_name + f'/params.pkl'
        if device.type == 'cpu':
            state = torch.load(params_fname, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(params_fname)
        model.load_state_dict(state)

    def infer(x):
        model.eval()
        x = x.astype(np.int64).reshape(1, -1)  # Append batch axis (100,) -> (1, 100)
        # print(x)
        # for BERT counting
        shop_len = np.count_nonzero(x)
        a = np.insert(x, 0, CLR)
        x = torch.as_tensor(a, device=device)
        segment_label = torch.as_tensor(np.zeros_like(a).astype(np.long), device=device)
        pred = model(x.unsqueeze(0),segment_label.unsqueeze(0)).detach().cpu().numpy()[0]
        print(pred)
        pred = np.argmax(pred)
        return pred

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

#############################################################

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--num-labeled', type=int, default=2000, help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total-steps', default=2000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=50, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num-classes', default=10, type=int, help='number of classes')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.001, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.001, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0.00005, type=float, help='train weight decay')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=20, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=32, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=0.00005, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0.0005, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0.9, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=16, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.9, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")


parser.add_argument('--mode', type=str, default='train', help='submit?????? ???????????? test??? ???????????????.')
parser.add_argument('--iteration', type=str, default='0',
                  help='fork ???????????? ??????????????? ?????????????????? ???????????????. ??????????????? ????????? ????????? ????????? wall time ??? model ??? ???????????????.')
parser.add_argument('--pause', type=int, default=0, help='model ??? load ?????? 1??? ???????????????.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--output_dir', type=str, default='result')
parser.add_argument('--fc_dim', type=int, default=128)
parser.add_argument('--epoch_max', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--load_model', type=str, default=None)

parser.add_argument('--loss_function', type=str, default=None)  # custom.
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--model', type=str, default='baseline')
parser.add_argument('--log_freq', type=int, default=50)
parser.add_argument('--device')
parser.add_argument('--valid_ratio', type = float , default= 0.1 )
parser.add_argument('--freeze_bert', type = bool , default = True )
parser.add_argument('--hidden_dim', type = int, default= 768)
parser.add_argument('--n_layers', type = int , default = 6 )
parser.add_argument('--attn_heads',  type = int  , default = 2 )
parser.add_argument("--data_aug" , type = bool ,default = False)
parser.add_argument("--k_fold" ,type = int , default = None)



def DATA_GETTERS(args ,  all_x , train_x , train_y  ) :
    original_train_x = np.copy(train_x)
    original_train_y = np.copy(train_y)

    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=args.valid_ratio, shuffle=True)
    labeled_dataset = dataset.BertDatatset_classification(train_x , train_y , istrain = True ,masking  = False )

    unlabeled_dataset  = dataset.BertDatatset_unlabeled(all_x = all_x ,shuffle= True )

    test_dataset =  dataset.BertDatatset_classification(valid_x , valid_y , istrain = False ,masking  = False )

    finetune_dataset  =  dataset.BertDatatset_classification(train_x , train_y , istrain = True ,masking  = False )


    return labeled_dataset , unlabeled_dataset , test_dataset , finetune_dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(args, labeled_loader, unlabeled_loader, test_loader, finetune_dataset,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    # for author's code formula
    # moving_dot_product = torch.empty(1).to(args.device)
    # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    # nn.init.uniform_(moving_dot_product, -limit, limit)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            datas_l, targets , segment_labels = labeled_iter.next()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            datas_l , targets , segment_labels = labeled_iter.next()

        try:
            datas_uw, datas_us ,  segment_labels2 , segment_labels3  = unlabeled_iter.next()
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            datas_uw, datas_us , segment_labels2 , segment_labels3  = unlabeled_iter.next()

        data_time.update(time.time() - end)

        datas_l = datas_l.to(args.device)
        datas_uw = datas_uw.to(args.device)
        datas_us = datas_us.to(args.device)
        targets = targets.to(args.device)
        segment_labels = segment_labels.to(args.device)
        segment_labels2 = segment_labels2.to(args.device)
        segment_labels3 = segment_labels3.to(args.device)

        with amp.autocast(enabled=args.amp):
            batch_size = datas_l.shape[0]
            t_images = torch.cat((datas_l, datas_uw, datas_us))
            t_logits = teacher_model(t_images , torch.cat((segment_labels, segment_labels2 , segment_labels3)))
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits

            t_loss_l = criterion(t_logits_l, targets)

            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((datas_l, datas_us))
            s_logits = student_model(s_images , torch.cat((segment_labels , segment_labels3 )))
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label)

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)

        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()

        if args.ema > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(datas_l , segment_labels)

            # print(targets[0])
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)
            print(s_loss_l_new, s_loss_l_old)
            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            # dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            dot_product = s_loss_l_new - s_loss_l_old
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_en
            # test
            # t_loss_mpl = torch.tensor(0.).to(args.device)
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            # wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                # wandb.log({"train/1.s_loss": s_losses.avg,
                #            "train/2.t_loss": t_losses.avg,
                #            "train/3.t_labeled": t_losses_l.avg,
                #            "train/4.t_unlabeled": t_losses_u.avg,
                #            "train/5.t_mpl": t_losses_mpl.avg,
                #            "train/6.mask": mean_mask.avg})

                test_model = avg_student_model if avg_student_model is not None else student_model
                test_loss, top1, top5 = evaluate(args, test_loader, test_model, criterion)

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                args.writer.add_scalar("test/acc@5", top5, args.num_eval)
                # wandb.log({"test/loss": test_loss,
                #            "test/acc@1": top1,
                #            "test/acc@5": top5})

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5
                    bind_model(avg_student_model if avg_student_model is not None else student_model , args.device)
                    nsml.save('best_acc{}'.format(int(top1)))
                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if args.local_rank in [-1, 0]:
        args.writer.add_scalar("result/test_acc@1", args.best_top1)


    # finetune
    del t_scaler, t_scheduler, t_optimizer, teacher_model, labeled_loader, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])

    finetune(args, finetune_dataset, test_loader, student_model, criterion)

    return teacher_model , student_model


def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets , segment_labels) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            segment_labels = segment_labels.to(args.device)
            with amp.autocast(enabled=args.amp):
                outputs = model(images, segment_labels)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            losses.update(loss.item(), batch_size)

            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")

        test_iter.close()
        return losses.avg, top1.avg, top5.avg


def finetune(args, finetune_dataset, test_loader, model, criterion):
    model.drop = nn.Identity()
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        finetune_dataset,
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    optimizer = optim.Adam(model.parameters(),
                          lr=args.finetune_lr,
                          weight_decay=args.finetune_weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")
    best_loss = 10000
    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        epoch_loss = 0
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets , segment_labels) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            segment_labels = segment_labels.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                outputs = model(images , segment_labels)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss, top1, top5 = evaluate(args, test_loader, model, criterion)
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            args.writer.add_scalar("finetune/acc@1", top1, epoch)
            args.writer.add_scalar("finetune/acc@5", top5, epoch)

            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top5 = top5

            if epoch_loss < best_loss  :
                best_loss = epoch_loss
                bind_model(model,args.device)
                nsml.save("fine{}".format(epoch))
            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'best_top1': args.best_top1,
                'best_top5': args.best_top5,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("result/finetune_acc@1", args.best_top1)

    return


def main():


    args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top5 = 0.

    if args.local_rank != -1:
        args.gpu = args.local_rank
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.gpu = 0
        args.world_size = 1

    args.device = torch.device('cuda', args.gpu)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")


    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # labeled_dataset, unlabeled_dataset, test_dataset, finetune_dataset = DATASET_GETTERS[args.dataset](args)

    labeled_dataset  , unlabeled_dataset  , test_dataset , finetune_dataset = DATA_GETTERS(args = args , all_x = all_x , train_x = train_x , train_y  =train_y )

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)

    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)

    # if args.dataset == "cifar10":
    #     depth, widen_factor = 28, 2
    # elif args.dataset == 'cifar100':
    #     depth, widen_factor = 28, 8

#######################################################################################################################
    VOCAB_SIZE = 5003

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.model == 'EDA':

        teacher_model = ModelEMA(model=BERT.BERTClassifer(bert=BERT.get_BERT(args), args=args, hidden=args.hidden_dim,
                                                          freeze=args.freeze_bert).to(args.device))

    elif args.model == 'essemble' :

        techer_model = essemble_model(args = args)

    else:
        teacher_model = BERT.BERTClassifer(bert=BERT.get_BERT(args), args=args, hidden=args.hidden_dim,
                                           freeze=args.freeze_bert).to(args.device)
    if args.load_model is not None:
        session, ckpt = args.load_model.split('_')
        bind_model(teacher_model, args.device)
        nsml.load(checkpoint=ckpt, session='KR96419/airush2022-2-5/' + session)
        nsml.save(args.load_model + 'classifier')
        print('saved')

    bind_model(teacher_model, args.device)

    if args.pause:
        nsml.paused(scope=locals())
    ######################################################################################################################

    ## ???????????? ????????? ?????? ?????? ???????????????!
    # args.fc_dim *= 2
    if args.model == 'EDA':
        student_model = BERT.BERTClassifer(bert=teacher_model.module.bert, args=args, hidden=args.hidden_dim,
                                           freeze=args.freeze_bert).to(args.device)
    else:
        student_model = BERT.BERTClassifer(bert=teacher_model.bert, args=args, hidden=args.hidden_dim,
                                           freeze=args.freeze_bert).to(args.device)

    # student_model = WideResNet(num_classes=args.num_classes,
    #                            depth=depth,
    #                            widen_factor=widen_factor,
    #                            dropout=0,
    #                            dense_dropout=args.student_dropout)


    if args.local_rank == 0:
        torch.distributed.barrier()

    # logger.info(f"Model: WideResNet {depth}x{widen_factor}")

    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")

    teacher_model.to(args.device)
    student_model.to(args.device)
    avg_student_model = None
    if args.ema > 0:
        avg_student_model = ModelEMA(student_model, args.ema)

    criterion = create_loss_fn(args)

    no_decay = ['bn']
    teacher_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_optimizer = optim.SGD(teacher_parameters,
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    ###########################################################################################
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")
    ################################################################################################


    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    if args.finetune:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_scheduler, s_optimizer
        finetune(args, finetune_dataset, test_loader, student_model, criterion)
        return

    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        evaluate(args, test_loader, student_model, criterion)
        return

    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(args, labeled_loader, unlabeled_loader, test_loader, finetune_dataset,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler)
    return


if __name__ == '__main__':
    main()
