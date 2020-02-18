from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension('lpc.coding', ['src/coding.pyx']),
    Extension('lpc.functions', ['src/functions.pyx'])
]

setup(
    name='lpc',
    install_requires=[
        'numpy',
        'scipy',
        'sounddevice',
        'cython'
    ],
    ext_modules=cythonize(extensions)
)
