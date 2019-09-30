import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="tacas",
    version="0.0.1",
    author="Simone Silvetti",
    author_email="simone.silvetti@gmail.com",
    description=("PyCheck"),
    license="BSD",
    keywords="boh",
    url="http://",
    packages=['pycheck.tessellation', 'pycheck.regressor'],
    install_requires=['numpy>=1.13.1','scikit-learn>=0.19.0','scipy>=0.19.1','matplotlib>=2.0.2','pyDOE>=0.3.8','numba>=0.35.0',],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 0 - Alpha",
        "Topic :: Utilities",
        "License :: BSD License",
    ],
)
