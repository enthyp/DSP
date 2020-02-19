from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension('lpc.coding', ['lpc/coding.pyx'], include_dirs=['lpc/']),
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
    ext_modules=cythonize(extensions),
    zip_safe=False,
    package='lpc',
    package_data={
        'lpc/coding': ['lpc/functions.pxd'],
    }
)
