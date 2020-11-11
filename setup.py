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
      author='Mikhail Kurenkov',
      author_email='kurenkovrobotics@gmail.com',
      package_dir={},
      packages=["mvae", "mvae.data", "mvae.models", "mvae.utils", "mvae.models.factory"],
      install_requires=install_requires
      )
