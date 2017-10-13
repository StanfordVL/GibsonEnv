from distutils.core import setup
from Cython.Build import cythonize

setup(name="transfer", ext_modules=cythonize('transfer.pyx'),)