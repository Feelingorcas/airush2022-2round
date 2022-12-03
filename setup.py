#nsml: registry.navercorp.com/nsml/torch-cv/torch-1.3keras2.3

from distutils.core import setup


setup(
    name='MAI_Baseline',
    version='1',
    description='MAI_Baseline',
    install_requires=['wandb' , 'torch ==1.7.0' , 'torchvision == 0.8.0' , 'requests'
    ]
)
