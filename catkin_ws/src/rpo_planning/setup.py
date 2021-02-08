#!/usr/bin/python
from setuptools import find_packages
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

packages = find_packages('src')
print('packages: ', packages)
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'rpo_planning' or p.startswith('rpo_planning.')


# setup_args = generate_distutils_setup(
# 	packages=['skills'],
# 	package_dir={'': 'src'}
# )

setup_args = generate_distutils_setup(
	packages=packages,
	package_dir={'': 'src'}
)
setup(**setup_args)

