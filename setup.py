# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import os
import setuptools
from setuptools import setup
from lips import get_version

#__version__ = '0.1.0'
#print(get_version("__init__.py"))


def get_data_files(directory):
    my_list = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            my_list.append((dirpath, [os.path.join(dirpath,f)]))
    return my_list

pkgs = {
    "required": [
        "numpy==1.25.2",
        "scikit_learn",
        "tqdm",
        "matplotlib",
        "scipy==1.11.4",
        "six",
        "pathlib",
        "numba",
    ],
    "extras": {
        "recommended": [
            "grid2op==1.9.8", #>=1.7.2",
            "pybind11==2.8.1",
            "lightsim2grid==0.7.5", #>=0.7.0.post1",
            "leap-net==0.0.5",
            "protobuf==3.20.2",
            "pandapower==2.8.0",
            "pandas==1.5.3",
            "jupyter",
            "tensorflow==2.8.1",
            "torch==2.0.1",
            "imageio==2.34.0",
            "plotly==5.20.0"
        ],
        "docs": [
            "numpydoc>=0.9.2",
            "sphinx>=2.4.4",
            "sphinx-rtd-theme>=0.4.3",
            "sphinxcontrib-trio>=1.1.0",
            "autodocsumm>=0.1.13",
            "gym>=0.17.2"
        ],
        "codabench": [
            "filelock==3.7.1",
            "json2table==1.1.5",
            "loguru==0.6.0",
            "PyYAML==6.0",
            "tqdm==4.62.3"
        ],
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-html",
            "pytest-metadata",
            "ipykernel",
            "pylint",
            "pylint-exit",
            "jupytext"
        ]
    }
}

pkgs["extras"]["codabench"] += pkgs["extras"]["recommended"]
pkgs["extras"]["test"] += pkgs["extras"]["recommended"]
pkgs["extras"]["test"] += pkgs["extras"]["docs"]
pkgs["extras"]["test"] += pkgs["extras"]["codabench"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='lips-benchmark',
      version=get_version("__init__.py"),
      description='LIPS : Learning Industrial Physical Simulation benchmark suite',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='Physical system solver, augmented simulator, benchmarking',
      author='Milad Leyli-abadi',
      author_email='milad.leyli-abadi@irt-systemx.fr',
      url="https://github.com/IRT-SystemX/LIPS",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={
            # If any package contains *.txt or *.rst files, include them:
            "": ["*.ini"],
            },
      data_files=get_data_files("configurations"),#[("configurations/powergrid/benchmarks/", ["configurations/powergrid/benchmarks/l2rpn_case14_sandbox.ini"])],
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': []
     }
)
