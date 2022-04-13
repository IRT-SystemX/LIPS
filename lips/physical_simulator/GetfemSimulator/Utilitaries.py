#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import errno
import numpy as np
import pandas as pd

def CreateDirFromPath(path):
    """
    .. py:function:: CreateDirFromPath(path)

    Create directory if it does not already exist
    :param string path: path of the directory to create
    """
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def BasicDofNodesCoordinates(positions):
    """
    .. py:function:: BasicDofNodesCoordinates(positions)

    Reshaping nodes positions
    :param floats array positions: basic dof positions
    """

    coor_x=positions[:,0]
    coor_y=positions[:,1]
    coor_z=np.zeros_like(coor_y)
    return coor_x,coor_y,coor_z

def ComputeNorm2(field):
    """
    .. py:function:: ComputeNorm2(field)

    Compute euclidian norm of a field at each point
    :param float array field: vector valued field
    """
    field_x=field[0::2]
    field_y=field[1::2]
    norm2_disps=np.sqrt(field_x**2+field_y**2)
    return norm2_disps

def CSVWriterFromData(filename,data):
    myDataFrame=pd.DataFrame.from_dict(data)
    myDataFrame.to_csv(filename,index=False)

def VectorSolutionCsvFromData(filename,data,solFieldName):
    """
    .. py:function:: VectorSolutionCsvFromData(filename,data,solFieldName)

    Create csv file in the so-called VectorSolution output format, that is to say:
    Ux1, Uy1, x1, y1, 
    Ux2, Uy2, x2, y2,
    Ux3, Uy3, x3, y3, 
    Ux4, Uy4, x4, y4, 

    :param pointer filename: csv file object
    :param float array by string data: data from solver
    :param string solFieldName: output field name
    """

    positions = data['Positions']
    coor_x, coor_y, coor_z = BasicDofNodesCoordinates(positions)

    posSolData = {solFieldName[0]:data[solFieldName[0]],solFieldName[1]:data[solFieldName[1]],'x':coor_x, 'y':coor_y, 'z':coor_z}
    CSVWriterFromData(filename=filename,data=posSolData)

def ScalarSolutionCsvFromData(filename,data,solFieldName):
    """
    .. py:function:: ScalarSolutionCsvFromData(filename,data,solFieldName)

    Create csv file in the so-called ScalarSolution output format, that is to say:
    N(U1), x1, y1, z1
    N(U2), x2, y2, z2
    N(U3), x3, y3, z3
    N(U4), x4, y4, z4

    :param pointer filename: csv file object
    :param float array by string data: data from solver
    :param string solFieldName: output field name
    """
    positions=data['Positions']
    solField=data[solFieldName[0]]
    coor_x,coor_y,coor_z=BasicDofNodesCoordinates(positions)
    norm2_sols=ComputeNorm2(solField)

    posSolData = {'norm'+str(solFieldName):norm2_sols,'x':coor_x, 'y':coor_y, 'z':coor_z}
    CSVWriterFromData(filename=filename,data=posSolData)

def ScalarFieldCsvFromData(filename,data,solFieldName):
    """
    .. py:function:: ScalarSolutionCsvFromData(filename,data,solFieldName)

    Create csv file in the so-called ScalarSolution output format, that is to say:
    U1, x1, y1, z1
    U2, x2, y2, z2
    U3, x3, y3, z3
    U4, x4, y4, z4

    :param pointer filename: csv file object
    :param float array by string data: data from solver
    :param string solFieldName: output field name
    """
    positions,solField=data['Positions'],data[solFieldName]
    coor_x,coor_y,coor_z=BasicDofNodesCoordinates(positions)

    posSolData = {str(solFieldName):solField,'x':coor_x, 'y':coor_y, 'z':coor_z}
    CSVWriterFromData(filename=filename,data=posSolData)

def WriteDataInCsv(outputformat,filename,data,fieldName):
    """
    .. py:function:: WriteDataInCsv(outputformat,filename,data,fieldName)

    Create csv file in the chosen output format
    :param string outputformat: format output name
    :param string filename: output file name
    :param float array by string data: data from solver
    :param string fieldName: output field name
    """
    formatfuncbytype={"VectorSolution":VectorSolutionCsvFromData,"ScalarSolution":ScalarSolutionCsvFromData,"ScalarField":ScalarFieldCsvFromData}
    try:
        formatfuncbytype[outputformat](filename,data,fieldName)
    except KeyError:
        print("Not recognized format: "+str(outputformat))


def CheckIntegrity_WriteVectorSolutionDataInCsv(filename,data,dataNames):
    WriteDataInCsv("VectorSolution",filename,data,dataNames)
    myData=np.genfromtxt(filename,skip_header=1,delimiter=',')
    Ux=data['Ux'].reshape((data['Ux'].shape[0],1))
    Uy=data['Uy'].reshape((data['Uy'].shape[0],1))
    zCoord=np.zeros((data['Uy'].shape[0],1))
    checkData=np.concatenate((Ux,Uy,data['Positions'],zCoord),axis=1)
    assert np.linalg.norm(checkData-myData)<1e-8

def CheckIntegrity_WriteScalarSolutionDataInCsv(filename,data,dataNames):
    WriteDataInCsv("ScalarSolution",filename,data,dataNames)
    myData=np.genfromtxt(filename,skip_header=1,delimiter=',')
    myDisp=ComputeNorm2(data['disp']).reshape((data['disp'].shape[0]//2,1))
    zCoord=np.zeros((data['disp'].shape[0]//2,1))
    checkData=np.concatenate((myDisp,data['Positions'],zCoord),axis=1)
    assert np.linalg.norm(checkData-myData)<1e-8

def CheckIntegrity_WriteFieldDataInCsv(filename,data,dataNames):
    WriteDataInCsv("ScalarField",filename,data,dataNames)
    myData=np.genfromtxt(filename,skip_header=1,delimiter=',')
    myMult=data['mult'].reshape((data['mult'].shape[0],1))
    zCoord=np.zeros((data["mult"].shape[0],1))
    checkData=np.concatenate((myMult,data['Positions'],zCoord),axis=1)
    assert np.linalg.norm(checkData-myData)<1e-8

def CheckIntegrity():
    positions=np.random.rand(1000,2)
    displacements=np.random.rand(2000)
    data = {'Positions': positions}
    data['Ux'] = displacements[0::2]
    data['Uy'] = displacements[1::2]
    dataNames=['Ux', 'Uy']
    CheckIntegrity_WriteVectorSolutionDataInCsv(filename="file1.csv",data=data,dataNames=dataNames)
    data = {'Positions': positions}
    data['disp'] = displacements
    dataNames=['disp']
    CheckIntegrity_WriteScalarSolutionDataInCsv(filename="file2.csv",data=data,dataNames=dataNames)

    multipliers=np.random.rand(1000)
    data = {'Positions': positions,'mult':multipliers}
    dataNames='mult'
    CheckIntegrity_WriteFieldDataInCsv(filename="file3.csv",data=data,dataNames=dataNames)
    return "OK"

if __name__ =="__main__":
    CheckIntegrity()