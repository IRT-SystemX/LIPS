
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from lips.physical_simulator.GetfemSimulator.Utilitaries import WriteDataInCsv

import getfem as gf
gf.util_trace_level(1)

def ExportFieldInGmsh(filename,mfu,U,mfField,field,fieldName):
    mfu.export_to_pos(filename, mfu, U, 'Displacement',mfField,field,fieldName) 

def ExportFieldInVTK(filename,mfu,U,mfField,field,fieldName):
    mfu.export_to_vtk(filename, mfu, U, 'Displacement',mfField,field,fieldName) 

def ExportSingleFieldInGmsh(filename,mfField,field,fieldName):
    mfField.export_to_pos(filename, mfField,field,fieldName) 

def ExportSingleFieldInVTK(filename,mfField,field,fieldName):
    mfField.export_to_vtk(filename, mfField,field,fieldName) 

def ExportPrimalSolutionInGmsh(filename,feSpace,solutions):
    feSpace.export_to_pos(filename, feSpace, solutions, 'Displacement')

def ExportPrimalSolutionInVTK(filename,feSpace,solutions):
    feSpace.export_to_vtk(filename, feSpace, solutions, 'Displacement')

def ExportTransientPrimalSolutionInGmsh(filename,feSpace,solutions):
    fieldToExport=[[solution,'Displacement'+str(solId)] for solId,solution in enumerate(solutions)]
    fieldToExport=[item for sublist in fieldToExport for item in sublist]
    feSpace.export_to_pos(filename, feSpace, *fieldToExport)

def ExportTransientPrimalSolutionInVTK(filename,feSpace,solutions):
    fieldToExport=[[solution,'Displacement'+str(solId)] for solId,solution in enumerate(solutions)]
    fieldToExport=[item for sublist in fieldToExport for item in sublist]
    feSpace.export_to_vtk(filename, feSpace, *fieldToExport)

def ExportTransientPrimalSolutionInVTKSplitFiles(filename,feSpace,solutions):
    filenamePrefix=filename.split(".")[0]
    for solId,solution in enumerate(solutions):
        outputFile=filenamePrefix+str(solId)+".vtk"
        feSpace.export_to_vtk(outputFile, feSpace, solution,'Displacement')

def ExportTransientPrimalDualSolutionInGmsh(filename,feSpaces,solutions,contactBoundary):
    mfdisp,mfmult=feSpaces
    disps,mults=solutions
    refIndices=mfmult.basic_dof_on_region(contactBoundary)
    dispFieldToExport,multFieldToExport=[],[]
    for solId,(disp,mult) in enumerate(zip(disps,mults)):
        dispFieldToExport.append(disp)
        dispFieldToExport.append('Displacement'+str(solId))

        extendedLambda=np.zeros(mfmult.nbdof())
        extendedLambda[refIndices]=-mult
        multFieldToExport.append(mfmult)
        multFieldToExport.append(extendedLambda)
        multFieldToExport.append('Multiplier'+str(solId))

    mfdisp.export_to_pos(filename, mfdisp, *dispFieldToExport, *multFieldToExport)

def ExportTransientPrimalDualSolutionInVtk(filename,feSpaces,solutions,contactBoundary):
    filenamePrefix=filename.split(".")[0]
    mfdisp,mfmult=feSpaces
    disps,mults=solutions
    refIndices=mfmult.basic_dof_on_region(contactBoundary)
    for solId,(disp,mult) in enumerate(zip(disps,mults)):
        extendedLambda=np.zeros(mfmult.nbdof())
        extendedLambda[refIndices]=-mult
        
        outputFile=filenamePrefix+str(solId)+".vtk"
        mfdisp.export_to_vtk(outputFile, mfdisp, disp,'Displacement',mfmult,extendedLambda,'Multiplier')

def ExportSolutionInGmsh(filename,mflambda,lambdaC,contactZone,mfu,U,VM=None,mfVM=None):
    refIndices=mflambda.basic_dof_on_region(contactZone)
    extendedLambda=np.zeros(mflambda.nbdof())
    extendedLambda[refIndices]=-lambdaC

    if VM:
        mfu.export_to_pos(filename, mfVM, VM, 'Von Mises Stress', mfu, U, 'Displacement',mflambda,extendedLambda,"Lagrange multiplier") 
    else:
        mfu.export_to_pos(filename, mfu, U, 'Displacement',mflambda,extendedLambda,"Lagrange multiplier") 

def ExportSolutionInVTK(filename,mflambda,lambdaC,contactZone,mfu,U,VM=None,mfVM=None):
    refIndices=mflambda.basic_dof_on_region(contactZone)
    extendedLambda=np.zeros(mflambda.nbdof())
    extendedLambda[refIndices]=-lambdaC

    if VM:
        mfu.export_to_vtk(filename, mfVM, VM, 'Von Mises Stress', mfu, U, 'Displacement',mflambda,extendedLambda,"Lagrange multiplier") 
    else:
        mfu.export_to_vtk(filename, mfu, U, 'Displacement',mflambda,extendedLambda,"Lagrange multiplier") 


def ExportSolutionInCSV(filename,outputformat,mfu,displacements):
    #Nodes original position
    allPositions=mfu.basic_dof_nodes()
    positions=np.transpose(allPositions)[1::2]
    data = {'Positions': positions}

    #Nodes displacements
    if outputformat == 'VectorSolution':
        data['Ux'] = displacements[0::2]
        data['Uy'] = displacements[1::2]
        dataNames=['Ux', 'Uy']
    else:
        data['Displacements'] = displacements
        dataNames='Displacements'
    WriteDataInCsv(outputformat, filename, data, dataNames)

def ExportTransientSolutionInCSV(filename,outputformat,mfu,solutions):
    filenamePrefix=filename.split(".")[0]
    nbSamples=len(solutions)
    maxIdDigits=len(str(nbSamples))+1
    for solId,solution in enumerate(solutions):
        fileId=str(solId).zfill(maxIdDigits)
        outputFile=filenamePrefix+fileId+".csv"
        ExportSolutionInCSV(outputFile,outputformat,mfu,solution)

def ExportNodalStressFieldInCSV(filename,outputformat,mfu,fieldVal):
    allPositions=mfu.basic_dof_nodes()
    positions=np.transpose(allPositions)[1::2]
    data = {'Positions': positions}

    #Nodes displacements
    if outputformat == 'VectorSolution':
        data['Fx'] = fieldVal[0::2]
        data['Fy'] = fieldVal[1::2]
        dataNames=['Fx', 'Fy']
    else:
        data['NodalStressField'] = fieldVal
        dataNames=['NodalStressField']
    WriteDataInCsv(outputformat,filename,data,dataNames)

def GetPositionFromZone(feSpace,zone):
    refIndices=feSpace.basic_dof_on_region(zone)
    allPositions=feSpace.basic_dof_nodes()
    positions=np.transpose(allPositions)[refIndices]
    return positions

def ExportNDofMultipliersInCSV(filename,mflambda,multipliers,contactZone):
    #Nodes original position
    positions=GetPositionFromZone(feSpace=mflambda,zone=contactZone)[1::2]
    #Multipliers
    data={'Positions':positions,
          'Multipliers_x':multipliers[0::2],
          'Multipliers_y':multipliers[1::2]}
    dataNames=['Multipliers_x','Multipliers_y']
    WriteDataInCsv("VectorSolution",filename,data,dataNames)

def ExportUnilateralMultipliersInCSV(filename,mflambda,multipliers,contactZone):
    #Nodes original position
    positions=GetPositionFromZone(feSpace=mflambda,zone=contactZone)
    #Multipliers
    data={'Positions':positions,'Multipliers':multipliers}
    WriteDataInCsv("ScalarField",filename,data,'Multipliers')