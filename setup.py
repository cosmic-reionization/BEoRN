'''
Created on 21 December 2022
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='beorn',
      version='0.1.0',
      author='T. Schaeffer, S. K. Giri, A. Schneider',
      author_email='sambit.giri@su.se',
      packages=find_packages("src"),
      package_dir={"": "src"},
      package_data={'beorn': ['input_data/*','files/*']},
      # install_requires=['numpy', 'scipy', 'matplotlib',
      #                   'pytest','tools21cm', 'astropy'],
      install_requires=requirements,
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
)
