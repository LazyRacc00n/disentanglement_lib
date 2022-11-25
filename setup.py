# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for disentanglement_lib."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='disentanglement_lib',
    version='1.5',
    description=('Library for research on disentangled representations.'),
    author='DisentanglementLib Authors',
    author_email='no-reply@google.com',
    url='http://github.com/google-research/disentanglement_lib',
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        'bin/dlib_aggregate_results',
        'bin/dlib_reproduce',
        'bin/dlib_reason',
        'bin/dlib_visualize_dataset',
        'bin/dlib_evaluate',
        'bin/dlib_udr',
        'bin/dlib_postprocess',
        'bin/dlib_train',
        'bin/dlib_visualize_dataset',
        'bin/dlib_visualize_model',
        'bin/dlib_tests',
        'bin/dlib_download_data',
        'bin/dlib_reproduce_jmlr',
        'bin/dlib_reproduce_semi_supervised',
        'bin/dlib_reproduce_weakly_supervised',
        'bin/dlib_train_semi_supervised',
        'bin/dlib_train_weakly_supervised',
    ],
    install_requires=[
        'gast==0.2.2',
	'protobuf==3.19.6',
        'future==0.18.2',
        'imageio==2.22.1',
        'gin-config==0.2.0',
        'scikit-learn==1.0.2',
        'numpy==1.16.4',
        'pandas==1.1.5',
        'simplejson==3.17.6',
        'six==1.16.0',
        'matplotlib==3.4.0',
        'pillow==9.2.0',
        'scipy==1.5.0',
        'tensorflow_hub==0.9.0',
        'tensorflow_probability==0.7.0',
    ],
    extras_require={
        'tf': ['tensorflow==1.14'],
        'tf_gpu': ['tensorflow-gpu==1.14'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning disentanglement learning',
)


# cudatoolkit == 10.0.130
# cudnn == 7.3.1
# 'saeborn==0.11.0',
