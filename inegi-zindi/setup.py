from setuptools import setup, find_packages

setup(
    name='Mistletoe-Detection',
    version='0.1.0',
    url='https://github.com/hucarlos08/mistletoe.git',
    author='Centro Geo',
    author_email='hcarlos@centrogeo.edu.mx',
    description='Description of my package',
    packages=find_packages(),
    install_requires=[
    'numpy', 
    'jupyter', 
    'jupyterlab',
    'torch',
    'torchvision',
    'matplotlib',
    'lightning',
    'rasterio',
    'wandb',
    'torchinfo', 
    'torchmetrics', 
    'pytorch_lightning',
                          ],
)