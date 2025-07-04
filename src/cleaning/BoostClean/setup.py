# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='activedetect',
    packages=find_packages(),
    version='0.1.4.post2',  
    description='A Library For Error Detection For Predictive Analytics',
    author='Sanjay Krishnan and Eugene Wu',
    author_email='sanjay@eecs.berkeley.edu',
    url='https://github.com/sjyk/activedetect/',
    download_url='https://github.com/sjyk/activedetect/tarball/0.1.4-2',
    keywords=['error', 'detection', 'cleaning'],
    classifiers=[],
    install_requires=[
        'numpy==1.16.1',        
        'scikit-learn==0.20.4',
        'gensim==3.8.3',       
        'usaddress==0.5.10',
        'scipy==1.2.3',
        'pandas==0.24.2',
        'tqdm==4.64.1'
    ]
)
