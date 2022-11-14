#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import griddata
import os

from lips.physical_simulator.GetfemSimulator.GetfemBricks.Utilities import Interpolate2DFieldOnSupport
from lips.physical_simulator.GetfemSimulator.Utilitaries import WriteDataInCsv,CreateDirFromPath

def WriteMeshToGridInterpolatedSolution(dirPath,phyProblem,FEMSolutions,targetSupport):
    CreateDirFromPath(dirPath)
    nbSamples=FEMSolutions.shape[0]
    maxIdDigits=len(str(nbSamples))+1
    for fieldId,field in enumerate(FEMSolutions):
        filePath=os.path.join(dirPath,"InterpolatedSol"+str(fieldId).zfill(maxIdDigits)+".csv")
        interpolatedField=FEMInterpolationOnSupport(phyProblem=phyProblem,originalField=field,targetSupport=targetSupport)
        WriteSingleMeshToGridInterpoledSolution(filePath=filePath,targetSupport=targetSupport,interpolatedField=interpolatedField)

def WriteSingleMeshToGridInterpoledSolution(filePath,targetSupport,interpolatedField):
    ux = interpolatedField[0::2]
    uy = interpolatedField[1::2]
    data = {'Positions': np.transpose(targetSupport), 'Ux': ux, 'Uy': uy}
    WriteDataInCsv('VectorSolution', filePath, data, ['Ux', 'Uy'])

def FEMInterpolationOnSupport(phyProblem,originalField,targetSupport,enableExtrapolation=False):
    model=phyProblem.model
    originalSupport=phyProblem.mesh
    interpolatedField=Interpolate2DFieldOnSupport(model=model,
                                                            originalSupport=originalSupport,
                                                            originalField=originalField,
                                                            targetSupport=targetSupport,
                                                            enableExtrapolation=enableExtrapolation)
    return interpolatedField

def InterpolateSolOnNodes(fieldSupport,fieldValue,targetSupport):
     interpolatedField = griddata(points=fieldSupport, values=fieldValue, xi=targetSupport, method='nearest')
     return interpolatedField

#################################Test#################################
import math
import numpy.testing as npt

from lips.physical_simulator.GetfemSimulator.GetfemWheelProblem import GetfemMecaProblem
from lips.physical_simulator.GetfemSimulator.GetfemBricks.MeshTools import GenerateWheelMesh
from lips.physical_simulator.GetfemSimulator.GetfemBricks.FeSpaces import GetUniqueBasicCoordinates

def CheckIntegrity_OuterDomainInterpolation():
    wheelDimensions=(8.,15.)
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    WPM=GetfemMecaProblem()
    WPM.mesh=GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=1,RefNumByRegion=refNumByRegion)
    WPM.refNumByRegion=refNumByRegion
    WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
    WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
    WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
    WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
    WPM.Preprocessing()
    WPM.BuildModel()
    state=WPM.RunProblem()
    u=WPM.GetSolution("disp")

    targetSupport=np.array([[0.0,100.0],[1.0,100.0]])
    uInterpol=FEMInterpolationOnSupport(phyProblem=WPM,originalField=u,targetSupport=targetSupport)
    npt.assert_equal(uInterpol[2],0.0)
    npt.assert_equal(uInterpol[3],0.0)
    WriteSingleMeshToGridInterpoledSolution(filePath="MyFile.csv",targetSupport=targetSupport,interpolatedField=uInterpol)
    return "ok"

class CheckIntegrity_InterpolateBackToMesh():
    def __init__(self):
        wheelDimensions=(8.,15.)
        self.refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
        self.mesh=GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=4,RefNumByRegion=self.refNumByRegion)

    def DefinePhyProblem(self):
        WPM=GetfemMecaProblem()
        WPM.refNumByRegion=self.refNumByRegion
        WPM.mesh=self.mesh
        WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
        WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
        WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
        WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
        WPM.Preprocessing()
        WPM.BuildModel()
        return WPM

    def test_interpolateGridSolutionBackToMesh(self):
        xpixels,ypixels,xsize,ysize,origin=128,128,32.0,32.0,(-16.0,0.0)
        coordX,coordY=np.meshgrid(np.arange(origin[0],origin[0]+xsize,xsize/xpixels),np.arange(origin[1],origin[1]+ysize,ysize/ypixels))
        gridPoints = np.vstack(list(zip(coordX.ravel(), coordY.ravel()))).transpose()
        targetSupport=gridPoints
        WPM=self.DefinePhyProblem()
        state=WPM.RunProblem()
        originalSolutionSolver=WPM.GetSolution("disp")
        interpolatedFieldGrid=FEMInterpolationOnSupport(phyProblem=WPM,originalField=originalSolutionSolver,targetSupport=targetSupport)
        originalSolution = np.column_stack((originalSolutionSolver[0::2],originalSolutionSolver[1::2]))

        interpolatedSol = np.column_stack((interpolatedFieldGrid[0::2],interpolatedFieldGrid[1::2]))
        interpolatedField={"Coords":targetSupport.transpose(),"Values":interpolatedSol}
        meshNodes=GetUniqueBasicCoordinates(WPM.feSpaces["disp"]).transpose()

        exteriorPointsRows = np.where(interpolatedSol[:,0] == 0.0) and np.where(interpolatedSol[:,1] == 0.0)
        interpolatedInteriorSol = np.delete(interpolatedSol, exteriorPointsRows, axis=0)
        interpolatedInteriorCoords=np.delete(targetSupport.transpose(), exteriorPointsRows, axis=0)
        interpolatedSolutionBack=InterpolateSolOnNodes(fieldSupport=interpolatedInteriorCoords,fieldValue=interpolatedInteriorSol,targetSupport=meshNodes)
        error=np.linalg.norm(interpolatedSolutionBack-originalSolution)
        print(error)

        myfieldOriginal=np.vstack((originalSolution[:,0],originalSolution[:,1])).ravel('F')
        WPM.ExportFieldInGmshWithFormat(filename='original.pos',dofpernodes=2,field=myfieldOriginal,elements_degree=2,fieldName="originalSolutionSolver")
        myfield=np.vstack((interpolatedSolutionBack[:,0],interpolatedSolutionBack[:,1])).ravel('F')
        WPM.ExportFieldInGmshWithFormat(filename='interpolated.pos',dofpernodes=2,field=myfield,elements_degree=2,fieldName="interpolatedSolutionSolver")
        return "ok"

    def test_interpolateMeshOnMesh(self):
        xpixels,ypixels,xsize,ysize,origin=128,128,32.0,32.0,(-16.0,0.0)
        coordX,coordY=np.meshgrid(np.arange(origin[0],origin[0]+xsize,xsize/xpixels),np.arange(origin[1],origin[1]+ysize,ysize/ypixels))
        gridPoints = np.vstack(list(zip(coordX.ravel(), coordY.ravel()))).transpose()
        WPM=self.DefinePhyProblem()
        state=WPM.RunProblem()
        originalSolutionSolver=WPM.GetSolution("disp")
        targetSupport=gridPoints
        originalSolution = np.column_stack((originalSolutionSolver[0::2],originalSolutionSolver[1::2]))
        meshNodes=GetUniqueBasicCoordinates(WPM.feSpaces["disp"]).transpose()
        interpolatedSolutionBack=InterpolateSolOnNodes(fieldSupport=meshNodes,fieldValue=originalSolution,targetSupport=meshNodes)
        error=np.linalg.norm(interpolatedSolutionBack-originalSolution)
        npt.assert_equal(error,0.0)
        return "ok"


class CheckIntegrity_InterpolateBackAndForth():
    def __init__(self):
        wheelDimensions=(8.,15.)
        self.refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
        self.mesh=GenerateWheelMesh(wheelDimensions=wheelDimensions,meshSize=4,RefNumByRegion=self.refNumByRegion)

    def DefinePhyProblem(self):
        WPM=GetfemMecaProblem()
        WPM.refNumByRegion=self.refNumByRegion
        WPM.mesh=self.mesh
        WPM.materials=[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]]
        WPM.sources=[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ]
        WPM.dirichlet=[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ]
        WPM.contact=[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
        WPM.Preprocessing()
        WPM.BuildModel()
        return WPM

    def test_InterpolationConsistency(self):
        WPM=self.DefinePhyProblem()
        WPM.RunProblem()
        u=WPM.GetSolution("disp")
        mfu=WPM.feSpaces["disp"]
        targetSupport=GetUniqueBasicCoordinates(mfu)
        uInterpol=FEMInterpolationOnSupport(phyProblem=WPM,originalField=u,targetSupport=targetSupport)
        npt.assert_allclose(u,uInterpol,atol=1e-12)
        return "ok"

    def test_WriteMeshToGridInterpolatedSols(self):
        WPM=self.DefinePhyProblem()
        WPM.RunProblem()
        u=WPM.GetSolution("disp")
        xpixels,ypixels,xsize,ysize,origin=64,64,32.0,32.0,(-16.0,0.0)
        coordX,coordY=np.meshgrid(np.arange(origin[0],origin[0]+xsize,xsize/xpixels),np.arange(origin[1],origin[1]+ysize,ysize/ypixels))
        targetSupport = np.vstack(list(zip(coordX.ravel(), coordY.ravel()))).transpose()
        WriteMeshToGridInterpolatedSolution(dirPath="MyDir",phyProblem=WPM,FEMSolutions=np.array([u,u]),targetSupport=targetSupport)
        return "ok"

    def test_BackToMesh(self):
        errors=[]
        WPM=self.DefinePhyProblem()
        for nbPixel in [64,128,256,512]:
            xpixels,ypixels,xsize,ysize,origin=nbPixel,nbPixel,32.0,32.0,(-16.0,0.0)
            coordX,coordY=np.meshgrid(np.arange(origin[0],origin[0]+xsize,xsize/xpixels),np.arange(origin[1],origin[1]+ysize,ysize/ypixels))
            gridPoints = np.vstack(list(zip(coordX.ravel(), coordY.ravel())))
            meshNodes=GetUniqueBasicCoordinates(WPM.feSpaces["disp"]).transpose()
            uInterpol=InterpolateSolOnNodes(fieldSupport=gridPoints,fieldValue=2*gridPoints,targetSupport=meshNodes)
            uTrue=2*meshNodes
            error=np.linalg.norm(uTrue-uInterpol)
            print("nbPixel: ",nbPixel," Error: ",error)
            errors.append(error)
        errors=np.array(errors)
        np.testing.assert_array_equal(errors, np.sort(errors)[::-1])
        return "ok"

def CheckIntegrity():
    interBackMesh=CheckIntegrity_InterpolateBackToMesh()
    interBackForth=CheckIntegrity_InterpolateBackAndForth()

    totest = [CheckIntegrity_OuterDomainInterpolation,
    interBackMesh.test_interpolateGridSolutionBackToMesh,
    interBackMesh.test_interpolateMeshOnMesh,
    interBackForth.test_InterpolationConsistency,
    interBackForth.test_WriteMeshToGridInterpolatedSols,
    interBackForth.test_BackToMesh
              ]

    for test in totest:
        res =  test()
        if  res.lower() != "ok" :
            return res

    return "OK"

if __name__ == '__main__':
    print(CheckIntegrity())