from setuptools import setup

setup(
    name='lpc',
    packages=['lpc'],
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'sounddevice'
    ]
)
