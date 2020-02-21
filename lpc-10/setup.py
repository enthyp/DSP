from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension('lpc.coding', ['lpc/coding.pyx']),
    Extension('lpc.functions', ['lpc/functions.pyx'])
]

setup(
    name='lpc',
    install_requires=[
        'numpy',
        'scipy',
        'sounddevice',
        'cython'
    ],
    ext_modules=cythonize(extensions, gdb_debug=True),
    zip_safe=False,
)
