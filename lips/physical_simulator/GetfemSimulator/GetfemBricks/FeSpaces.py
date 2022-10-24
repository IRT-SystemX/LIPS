
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import getfem as gf
gf.util_trace_level(1)

from lips.physical_simulator.GetfemSimulator.Utilitaries import BasicDofNodesCoordinates

#Definition Of FeSpaces
def DefineFESpaces(mesh, elements_degree, dof):
    mf = gf.MeshFem(mesh, dof)
    mf.set_classical_fem(elements_degree)
    return mf

def DefineDiscontinuousFESpace(mesh, elements_degree, dof):
    mfvm=gf.MeshFem(mesh,dof)
    mfvm.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(2,%d)' % (elements_degree,)))
    return mfvm

def DefineClassicalDiscontinuousFESpace(mesh,elements_degree,dof):
    mfp = gf.MeshFem(mesh,dof)
    mfp.set_classical_discontinuous_fem(elements_degree)
    return mfp

def GetBasicDof(mfu,regionTag=None,dofIds=None):
    if regionTag is not None:
        return mfu.basic_dof_on_region(regionTag)
    else:
        if dofIds is not None:
            return mfu.basic_dof_nodes(dofIds)
        return mfu.basic_dof_nodes()

def GetBasicCoordinates(mfu):
    allPositions=mfu.basic_dof_nodes()
    positions=np.transpose(allPositions)[1::2]
    return BasicDofNodesCoordinates(positions)

def GetUniqueBasicCoordinates(mfu):
    allPositions=mfu.basic_dof_nodes()
    coords=np.transpose(allPositions)[1::2].transpose()
    return coords

def GetNbDof(mfu):
    return mfu.nbdof()

def GetSolutionAsField(mfu,displacements):
    data={}
    #Nodes original position
    data["Positions"]=GetBasicCoordinates(mfu)
    #Nodes displacements
    data["Ux"] = displacements[0::2]
    data["Uy"] = displacements[1::2]
    return data