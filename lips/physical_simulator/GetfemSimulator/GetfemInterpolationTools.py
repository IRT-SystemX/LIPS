#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.interpolate import griddata
import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver

def FEMInterpolationOnSupport(phyProblem,originalField,targetSupport):
    interpolatedField=PhySolver.Interpolate2DFieldOnSupport(model=phyProblem.model,originalSupport=phyProblem.mesh,originalField=originalField,targetSupport=targetSupport)
    return interpolatedField

def InterpolateSolOnNodes(fieldSupport,fieldValue,targetSupport):
    interpolatedField = griddata(points=fieldSupport, values=fieldValue, xi=targetSupport, method='nearest')
    return interpolatedField
