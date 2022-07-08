"""
Usage:
    Introduce the sampling methods used to generate a space of parameters
Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""
 
import abc
import numpy as np
import pyDOE2 as doe
from typing import Union
import os

class Sampler(metaclass=abc.ABCMeta):
    """Base class for Sampler management
    This class represent a sampler, used to generate samples within a space parameters
    It merely generates the samples but it is the responsability of the derived classes to provide the actual sampling method.

    This is the base class of all Sampler in LIPS repository

    Parameters
    ----------
    name: dict
        the parameter space
    """
    def __init__(self,space_params):
        self.space_params=space_params
        self.sampling_output=[]
        self.sampling_type=""
    
    def generate_samples(self,nb_samples:int,sampler_seed:Union[None, int]=None):
        """Generate a sampling
        This function generate the samples.

        Parameters
        ----------
        nb_samples: int
            Number of samples wanted
        sampler_seed: Union[int, None]
            Seed for sampler for reproductibility
        """
        self.sampling_output=self._define_sampling_method(nb_samples=nb_samples,sampler_seed=sampler_seed)
        return self.sampling_output

    @abc.abstractmethod
    def _define_sampling_method(self,nb_samples:int,sampler_seed:Union[None, int]=None):
        """Define the sampling method
        This function is the actual implementation of the sampler considered. It must be redifened by the derived classes.

        Parameters
        ----------
        nb_samples: int
            Number of samples wanted
        sampler_seed: Union[int, None], optional
            Seed for sampler for reproductibility, by default None
        """
        pass

    def get_attributes_as_data(self,samples:Union[None, list]=None)-> dict:
        """Get parameter space discretization

        This functions retrieves the parameter space discretization arising from the sampler

        Parameters
        ----------
        samples : Union[None, list], optional
            samples list after sampling, by default None

        Returns
        -------
        dict
            samples of data

        Raises
        ------
        RuntimeError
            Check sampling parameters consistency (same parameter names for all samples)

        """
        if samples is None:
            samples=self.sampling_output

        fieldNum=[len(samples[0].keys()) for sample in samples]
        if fieldNum.count(fieldNum[0]) != len(fieldNum):
            raise RuntimeError("Samples do not have the same input parameters")

        value_by_input_attrib = {attribName: np.array([sample[attribName] for sample in samples]) for attribName in samples[0]}
        return value_by_input_attrib

    def save(self,path_out:str,samples:Union[None, list]=None):
        """Save samples in file

        This functions saves the samples in a file

        Parameters
        ----------
        path_out : str
            path for saving samples
        samples : Union[None, list], optional
            samples list after sampling, by default None

        """
        value_by_input_attrib = self.get_attributes_as_data(samples=samples)
        for attrib_name,data in value_by_input_attrib.items():
            np.savez_compressed(f"{os.path.join(path_out, attrib_name)}.npz", data=data)

    def __str__(self): 
        """
        It represents the sampler as a string.
        """
        s_info="Type of sampling: "+self.sampling_type+"\n"
        s_info+="Parameters\n"
        for param_name,paramVal in self.space_params.items():
            s_info+="\t"+str(param_name)+": "+str(paramVal)+"\n"
        return s_info 

    def __len__(self):
        """
        It provides the lenght of a sampler.
        """
        return len(self.sampling_output)


class LHSSampler(Sampler):
    """Sampler derived class
    This class represents a Latin Hypercube Sampler (LHS). It derived from the Sampler base class

    Parameters
    ----------
    name: dict
        the parameter space (mapping between parameter names and interval value for each parameter)
    """
    def __init__(self, space_params):
        super(LHSSampler,self).__init__(space_params=space_params) 
        self.sampling_type="LHSSampler"

    def _define_sampling_method(self,nb_samples,sampler_seed=None):
        """Define the LHS method
        This function is the actual implementation of the LHS method.

        Parameters
        ----------
        nb_samples: int
            Number of samples wanted
        sampler_seed: Union[int, None], optional
            Seed for sampler for reproductibility, by default None
        """
        space_params=self.space_params
        nfactor = len(space_params)
        self.vals =doe.lhs(nfactor, samples=nb_samples, random_state=sampler_seed, criterion="maximin")
        
        vals=np.transpose(self.vals)
        params_vect_by_name = {}
        for i,param_name in enumerate(space_params.keys()):
            min_val,max_val=space_params[param_name]
            params_vect_by_name[param_name] = min_val + vals[i]*(max_val - min_val)
        return list(map(dict, zip(*[[(k, v) for v in value] for k, value in params_vect_by_name.items()])))

if __name__ =="__main__":
    params={"params1":(21E6,22E6),"params2":(0.2,0.3)}
    sample_params=LHSSampler(space_params=params)
    test_params1=sample_params.generate_samples(nb_samples=2,sampler_seed=42)
    test_params2=sample_params.generate_samples(nb_samples=2,sampler_seed=42)
    assert test_params1==test_params2

    params={"params1":(21E6,22E6),"params2":(0.2,0.3)}
    sample_params=LHSSampler(space_params=params)
    nb_samples=5
    test_params=sample_params.generate_samples(nb_samples=nb_samples)
    assert len(test_params)==nb_samples
    print(sample_params)

