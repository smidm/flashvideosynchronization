# -*- coding: utf-8 -*-
from setuptools import setup
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
    name='flashvideosynchronization',
    version='1.0a1',
    author='Matěj Šmíd',
    url='https://github.com/smidm/flashvideosynchronization',
    author_email='m@matejsmid.cz',
    license='The MIT License',
    long_description='Sub-millisecond accurate multiple video synchronization using camera flashes.',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],    
    keywords='video synchronization',
    py_modules=['flashvideosynchronization'],
    install_requires=['git+https://github.com/smidm/imagesource.git', 'sklearn', 'joblib', 'numpy', 'matplotlib', 'scipy'],
    extras_require={'visualization': 'seaborn'},
)

