#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc

class BrickBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self._name=None

    @abc.abstractmethod
    def Build(self,problemParams:dict,brickParams:dict):
        pass

    def __call__(self,problemParams:dict,brickParams:dict):
        return self.Build(problemParams,brickParams)

    def __str__(self): 
        sInfo="Instance of "+type(self).__name__+"\n"
        sInfo+="brick name: "+str(self._name)+"\n"
        return sInfo