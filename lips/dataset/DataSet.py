# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import json

class DataSet():
    """
    Generic DataSet class allows to load 

    Attributes
    ----------
        experiment_name : ```str``
            the name of experiment for which a dataset is loaded or generated, which will be used to created related sub directories

             
    """

    def __init__(self):

        # number of samples in the dataset
        self.nb_samples = None

        # the data
        self.data = None

        # a flag to indicate if the dataset is available 
        self.is_available = False

        # the path where the data is located
        self.data_path = None

        # a tag is associated with a dataset to indicate its purpose and is used to create save directory 
        self.tag = None

    def read_from_file(self, path):
        """
        to load a dataset from disk or to load a saved dataset
        """
        pass

    def write_to_file(self, path):
        """
        to write the modified or generated dataset to disk
        """
        pass


    def generate(self, verbose=False):
        """
        to generate data from a simulator in the case of synthetic data
        """
        pass

    def save(self, path=None):
        """
        save the generated data and all the related meta data on disk
        """
        dir_out = os.path.join(path, "Data")
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)

        tag_dir = os.path.join(dir_out, self.tag)
        if not os.path.exists(tag_dir):
            os.mkdir(tag_dir)

        self.data_path = tag_dir

        self._save_metadata(tag_dir)

        return tag_dir


    def load(self, path=None):
        """
        load the already generated data
        """
        dir_in = os.path.abspath(path)
        if not os.path.exists(dir_in):
            raise RuntimeError("The indicated path does not exists")

        self.data_path = dir_in

        self._load_metadata(dir_in)

        return dir_in                

    def _save_metadata(self, path):
        res = self._get_metadata()
        json_nm = "metadata_DataSet.json"
        with open(os.path.join(path, json_nm), "w", encoding="utf-8") as f:
            json.dump(obj=res, fp=f)

    def _load_metadata(self, path):
        json_nm = "metadata_DataSet.json"
        with open(os.path.join(path, json_nm), "r", encoding="utf-8") as f:
            res = json.load(f)

        self.nb_samples = res["nb_samples"]
        self.data_path = res["data_path"]

        return res

    def _get_metadata(self):
        res = dict()
        res["nb_samples"] = self.nb_samples
        res["data_path"] = self.data_path

        return res