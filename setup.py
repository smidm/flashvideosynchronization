# -*- coding: utf-8 -*-
from setuptools import setup
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
    name='FlashVideoSynchronization',
    version='1.0b1',
    author='Matěj Šmíd',
    url='https://github.com/smidm/flashvideosynchronization',
    author_email='m@matejsmid.cz',
    license='The MIT License',
    long_description='Sub-millisecond accurate multiple video synchronization using camera flashes.',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],    
    keywords='video synchronization',
    py_modules=['flashvideosynchronization'],
    install_requires=['imagesource', 'sklearn', 'joblib', 'numpy', 'matplotlib', 'scipy'],
    extras_require=['seaborn'],
)

