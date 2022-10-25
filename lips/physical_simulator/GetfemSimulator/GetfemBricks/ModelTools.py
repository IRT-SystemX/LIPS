#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy import sparse

import getfem as gf
gf.util_trace_level(1)

import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

modelVarByPhyField={
    PFN.displacement:"u",
    PFN.contactMultiplier:"lambda",
    PFN.contactUnilateralMultiplier:"lambda"
}

def AddInitializedData(model,dataname,data):
    model.add_initialized_data(dataname, [data])

def DefineModel():
    model=gf.Model('real')
    return model

def AddVariable(model,var,mfvar,boundary=None):
    if boundary is None:
        model.add_fem_variable(var, mfvar)
    else:
        model.add_filtered_fem_variable(var, mfvar,boundary)

def AddFixedSizeVariable(model,variableName,variableSize):
    model.add_variable(variableName, variableSize)

def AddConstraintWithMultipliers(model,variableName,multVariableName,matConstraint,rhsConstraint):
    idBrick=model.add_constraint_with_multipliers(variableName,multVariableName,matConstraint,rhsConstraint)
    return idBrick


def PrintModel(model):
    model.variable_list()

def PrintBricks(model):
    model.brick_list()

def DeleteBoundaryCondition(model,BCId):
    model.delete_brick(BCId)
    try:
        model.delete_variable("mult_on_u")
    except:
        model.delete_variable("mult_on_u_2")

def DeleteVariable(model,variableName):
    model.delete_variable(variableName)

def CleanModel(model):     
    model.clear()

def GetNbDof(model):
    return model.nbdof()

def AssembleProblem(model):
    model.assembly()

def AddExplicitRHS(model,varName,explicitRHS):
    idBrick=model.add_explicit_rhs(varName, explicitRHS)
    return idBrick

def ExtractFullAssembledSystem(model):
    sparseTangentMatrix,rightHandSide=ExportSparseTangentMatrix(model),ExtractRHS(model)
    return sparseTangentMatrix,rightHandSide

def ExtractRHS(model):
    return model.rhs()

def ExportSparseTangentMatrix(model):
    tangentMatrix=ExtractTangentMatrix(model)
    data=tangentMatrix.csc_val()
    rows,nbval=tangentMatrix.csc_ind()
    nbdof=rows.shape[0]-1
    return sparse.csc_matrix((data, nbval, rows), shape=(nbdof, nbdof))

def ExtractTangentMatrix(model):
    return model.tangent_matrix()

def Solve(model,options):
    noisiness="noisiness" in options.keys() and options["noisiness"]
    options.pop('noisiness', None)
    solverArgs=list(options.items())
    solverArgs=[item for sublist in solverArgs for item in sublist]
    if noisiness:
        solverArgs+=['noisy']

    solverIt=model.solve(*tuple(solverArgs))
    return solverIt[0]<options["max_iter"]

def TwoStepsRollingSolve(problem):
    model,max_iter,max_residual=problem.model,problem.max_iter,problem.max_residual

    initVariablesName=[modelVarByPhyField[PFN.displacement],'lambda_D']
    initVariablesVal=GetModelVariables(model,initVariablesName)

    model.disable_variable('alpha_rot')
    print("Solve first step")

    solverOptions={"max_iter":max_iter,
            "max_res":max_residual,
            "noisiness":True
            }
    solverState=Solve(model,solverOptions)
    if not solverState:
        print("First step has failed")
        return solverState
    model.enable_variable('alpha_rot')

    SetModelVariables(model,initVariablesName,initVariablesVal)
    model.disable_variable('alpha_y')
    print("Solve second step")
    solverState=Solve(model,solverOptions)
    model.enable_variable('alpha_y')
    return solverState


def GetModelVariables(model,variableNames):
    sols=dict()
    for variableName in variableNames:
        sols[variableName]=GetModelVariableValue(model,variableName)
    return sols

def GetModelVariableValue(model,variableName):
    return model.variable(variableName)

def SetModelVariables(model,variableNames,variableValues):
    for variableName in variableNames:
        variableValue=variableValues[variableName]
        SetModelVariableValue(model,variableName,variableValue)

def SetModelVariableValue(model,variableName,variableValue):
    model.set_variable(variableName,variableValue)

def GetVariableDofInterval(model,variableModelName):
    varStart,varSize=model.interval_of_variable(variableModelName)
    return varStart,varStart+varSize

def ExportTangentMatrix(model):
    data=model.tangent_matrix().csc_val()
    rows,nbval=model.tangent_matrix().csc_ind()
    nbdof=rows.shape[0]-1
    return sparse.csc_matrix((data, nbval, rows), shape=(nbdof, nbdof))

def ExportRhs(model):
    return model.rhs()