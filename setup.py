from setuptools import setup, find_packages

setup(
    name='lpc_ib',
    version='1.0.0',
    description='Train neural networks to induce latent point collapse',
    author='Luigi Sbailò',
    author_email='luigi.sbailo@gmail.com',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.2',
        'numpy>=1.26',
        'scikit-learn>=1.2',
        'scipy>=1.11',
    ],
)
