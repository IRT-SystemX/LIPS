#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This file introduce the sampling methods used to generate a space of parameters 
import abc
import numpy as np
import csv
import pyDOE2 as doe

class Sampler(metaclass=abc.ABCMeta):
    def __init__(self,space_params):
        self.space_params=space_params
        self.sampling_output=[]
        self.sampling_name=""
    
    def generate_samples(self,nb_samples,sampler_seed=None):
        self.sampling_output=self._define_sampling_method(nb_samples=nb_samples,sampler_seed=sampler_seed)
        return self.sampling_output

    @abc.abstractmethod
    def _define_sampling_method(self,nb_samples,sampler_seed=None):
        pass

    def save_samples_in_file(self,filename,samples=None):
        if samples is None:
            samples=self.sampling_output

        fieldNum=[len(samples[0].keys()) for sample in samples]
        if fieldNum.count(fieldNum[0]) != len(fieldNum):
            raise RuntimeError("Samples do not have the same input parameters")

        with open(filename, mode='w') as csv_file:
            fieldnames = list(samples[0].keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for paramsSet in samples:
                writer.writerow(paramsSet)

    def __str__(self): 
        sInfo="Type of sampling: "+self.sampling_name+"\n"
        sInfo+="Parameters\n"
        for paramName,paramVal in self.space_params.items():
            sInfo+="\t"+str(paramName)+": "+str(paramVal)+"\n"
        return sInfo 

    def __len__(self):
        return len(self.sampling_output)


class LHSSampler(Sampler):
    def __init__(self, space_params):
        super(LHSSampler,self).__init__(space_params=space_params) 
        self.sampling_name="LHSSampler"

    def _define_sampling_method(self,nb_samples,sampler_seed=None):
        space_params=self.space_params
        nfactor = len(space_params)
        self.vals =doe.lhs(nfactor, samples=nb_samples, random_state=sampler_seed, criterion="maximin")
        
        vals=np.transpose(self.vals)
        paramsVectByName = {}
        for i,paramName in enumerate(space_params.keys()):
            minVal,maxVal=space_params[paramName]
            paramsVectByName[paramName] = minVal + vals[i]*(maxVal - minVal)
        return list(map(dict, zip(*[[(k, v) for v in value] for k, value in paramsVectByName.items()])))

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

