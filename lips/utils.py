# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import json
import numpy
from .benchmark import Benchmark
from typing import Callable


class NpEncoder(json.JSONEncoder):
    """
    taken from : https://java2blog.com/object-of-type-int64-is-not-json-serializable/
    """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        # if the object is a function, save it as a string
        if callable(obj):
            return str(obj)
        return super(NpEncoder, self).default(obj)


def get_path(root_path: str, benchmark: Benchmark):
    """get the path where the model or evaluation results should be saved

    Parameters
    ----------
    root_path : str
        _description_
    benchmark : Benchmark
        _description_
    """
    path = root_path / benchmark.env_name / benchmark.benchmark_name
    return path

class FunctionFactory:
    """General factory method to register functions (to be derived for specific purpose)
    """
    def __init__(self):
        self._creators = {}

    def register_function(self, function_name: str, creator: Callable, with_error=True):
        """Register new functions

        Parameters
        ----------
        function_name : str
            _description_
        creator : Callable
            _description_
        """
        if function_name in self._creators and with_error:
            raise (Exception ("Function with the name "+ str(function_name) +" already in the catalog") )
        self._creators[function_name] = creator

    def register_function_dict(self, function_dict: dict):
        """register a dictionary including functions

        Parameters
        ----------
        function_dict : dict
            _description_
        """
        for key_, fun_ in function_dict.items():
            self.register_function(key_, fun_)

    def get_function(self, function_name:str):
        """Get the required function

        Parameters
        ----------
        function_name : str
            _description_

        Returns
        -------
        Callable
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        creator = self._creators.get(function_name)
        if not creator:
            raise ValueError(function_name)
        return creator

    def get_all_names(self):
        return self._creators.keys()



class ClassFactory():
    """General factory method to register classes (to be derived for specific purpose)
    """
    def __init__(self):
        self._catalog = {}
        self._set_catalog = set()
    
    def register_class(self, class_name, class_type, constructor=None, with_error = True):
        """Register new classes

        Parameters
        ----------
        class_name : str
            _description_
        class_type : class (not an instance, the class itself)
            _description_
        constructor: callable to instantiate the class
            useful if arguments are required to instantiate the class
        with_error: boolean 
            allow catalog entry to be overwritten or not
        """
        if class_name in self._catalog and with_error:
           raise (Exception ("Class "+ str(name) +" already in the catalog") )
        self._catalog[class_name] = (class_type,constructor)
        if hasattr(self,"_set_catalog"):
            self._set_catalog.add( (class_name,class_type,constructor))

    def get_availables_combinations_for(self, class_name):
        """Retrieve combinations available for the required class

        Parameters
        ----------
        class_name : str
            _description_

        Returns
        -------
        list
            _description_

        """
        return [( obj,const) for key,obj,const in self._set_catalog if key == class_name]

    def create_instance(self,class_name,ops=None):
        """Instantiate the required class

        Parameters
        ----------
        class_name : str
            _description_
        ops : any
            arguments if constructor was provided when registering the class

        Returns
        -------
        class instance
            _description_

        """
        if class_name in self._catalog:
            class_type, classConstructor = self._catalog[class_name]
            if classConstructor is None:
                try:
                    classInstance = class_type()
                except Exception as e:
                    print("Error creating class (" +class_name+ "):" + str(class_type) + ". ")
                    raise(e)
            else:
                classInstance = classConstructor(ops)
            return classInstance

        raise(Exception("Unable to create object of type " + str(class_name) +"\n Possible object are :"+ str(list(self._catalog.keys()))  ))


def test_function_factory():
    import numpy as np
    class DummyFunctionFactory(FunctionFactory):
        def __init__(self):
            super(DummyFunctionFactory,self).__init__()

    myFunctionFactory=DummyFunctionFactory()
    myFunctionFactory.register_function(function_name="sum", creator=np.sum)
    myFunctionFactory.register_function(function_name="max", creator=np.max)
    myFunctionFactory.register_function(function_name="min", creator=np.min)

    for refuseFactoryModif in [True,False]:
        if refuseFactoryModif:
            try:    
                myFunctionFactory.register_function(function_name="min", creator=min,with_error=refuseFactoryModif)
            except:
                pass
            else:
                raise Exception()
        else:
            myFunctionFactory.register_function(function_name="min", creator=min,with_error=refuseFactoryModif)

    myMinFunction=myFunctionFactory.get_function("min")
    assert myMinFunction([10,7,1,16358])==1
    assert set(set(["sum","max","min","notDefined"] - myFunctionFactory.get_all_names()) )!=set()
    assert set(set(["sum","max","min"] - myFunctionFactory.get_all_names()) )==set()

def test_class_factory():
    class DummyClassFactory(ClassFactory):
        def __init__(self):
            super(DummyClassFactory,self).__init__()

    myClassFactory=DummyClassFactory()

    class DummyClassA():
        def __init__(self):
            self.dummyvariable=0
    myClassFactory.register_class(class_name="DummyA", class_type=DummyClassA)

    class DummyClassB():
        def __init__(self):
            self.dummyvariable=1
    myClassFactory.register_class(class_name="DummyB", class_type=DummyClassB)

    class DummyClassC():
        def __init__(self):
            self.dummyvariable=2
    myClassFactory.register_class(class_name="DummyC", class_type=DummyClassC)

    for className,expectedValue in zip(["DummyA","DummyB","DummyC"],[0,1,2]):
        test_instance=myClassFactory.create_instance(class_name=className)
        assert test_instance.dummyvariable==expectedValue

    for refuseFactoryModif in [True,False]:
        if refuseFactoryModif:
            try:    
                myClassFactory.register_class(class_name="DummyC", class_type=DummyClassC,with_error=refuseFactoryModif)
            except:
                pass
            else:
                raise Exception()
        else:
            myClassFactory.register_class(class_name="DummyC", class_type=DummyClassC,with_error=refuseFactoryModif)

    def DummyClassDBuilder(ops):
        return DummyClassD(ops)

    class DummyClassD():
        def __init__(self,myvalue):
            self.dummyvariable=myvalue
    myClassFactory.register_class(class_name="DummyD", class_type=DummyClassD, constructor=DummyClassDBuilder)

    combinationClassD=myClassFactory.get_availables_combinations_for(class_name="DummyD")
    assert len(combinationClassD)==1
    assert combinationClassD[0][1]!=None
    combinationClassC=myClassFactory.get_availables_combinations_for(class_name="DummyC")
    assert len(combinationClassC)==1
    assert combinationClassC[0][1]==None
    assert len(myClassFactory.get_availables_combinations_for(class_name="DummyF"))==0
    
    test_instance_builder=myClassFactory.create_instance(class_name="DummyD",ops=42)
    assert test_instance_builder.dummyvariable==42


if __name__ == '__main__':
    test_function_factory()
    test_class_factory()
