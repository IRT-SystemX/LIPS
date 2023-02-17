# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from lips.utils import FunctionFactory,ClassFactory

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

        