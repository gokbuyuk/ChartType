from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project is centered around the task of extracting data from images of 2D charts and plots. Examples are images of bar charts, line charts and scatter plots where one wants to obtain the underlying data that was used to generate the plots. This project was motivated by an ongoing Kaggle competition of 'Making Charts Accessible'',
    author='Precise',
    license='MIT',
)
