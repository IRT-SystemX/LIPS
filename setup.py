# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import setuptools
from setuptools import setup

pkgs = {
    "required": [
        "numpy",
        "scikit_learn",
        "tqdm",
        "matplotlib"
    ],
    "extras": {
        "recommended": [
            "grid2op>=1.6.2",
            "lightsim2grid>=0.5.3",
            "leap_net @ https://github.com/BDonnot/leap_net/tarball/master#egg=leap_net",
            "numba",
            "jupyter"
        ]
    }
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='lips',
      version='0.0.1',
      description='LIPS : Learning Industrial Physical Simulation benchmark suite',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='Physical system solver, augmented simulator, benchmarking',
      author='Milad Leyli-abadi',
      author_email='milad.leyli-abadi@irt-systemx.fr',
      url="https://github.com/Mleyliabadi/LIPS",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': []
     }
)
