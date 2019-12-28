from setuptools import setup

setup(
    name='cytokinin',
    version='0.0.1',
    description='Promotes data rooting and make your ML projects to flourish',
    author='Gianfrancesco Angelini',
    author_email='gian.angelini@hotmail.com',
    packages=['cytokine'],
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas>=0.25.3',
        'opencv-python>=4.1.0'
    ]
)
