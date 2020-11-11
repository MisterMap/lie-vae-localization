#!/usr/bin/env python

import os
from distutils.core import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='mvae_localization',
      version='0.1',
      description='MVAE Monte-Carlo robot localization project',
      author='kurenkovrobotics@gmail.com',
      author_email='',
      package_dir={},
      packages=["mvae", "mvae.data", "mvae.modules", "mvae.utils"],
      install_requires=install_requires
      )
