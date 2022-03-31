#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gmsh
import numpy as np

def GenerateWheelMeshFileStandardVersion(outputFile, wheelDimensions, meshSize):
    """PYO code using gmsh to generate a mesh"""
    gmsh.initialize()
    model_name=outputFile
    gmsh.model.add(model_name)
    cm = 1.
    center_x = 0 * cm
    center_y = 15 * cm
    rayon_1, rayon_2 = wheelDimensions
    Lc1 = meshSize
    Lc2 = 0.01
    # We start by defining some points and some lines. To make the code shorter we
    # can redefine a namespace:
    factory = gmsh.model.occ
    factory.addPoint(center_x, center_y, 0, Lc1, 1)
    # Circles
    factory.addCircle(center_x, center_y, 0, rayon_1, 10)  # instead of addCircleArc(2, 1, 2, 10)
    factory.addCircle(center_x, center_y, 0, rayon_2, 11)
    factory.addCurveLoop([10], 20)
    factory.addCurveLoop([11], 21)
    # Add surfaces
    factory.addPlaneSurface([21, 20], 30)
    factory.synchronize()
    gmsh.model.addPhysicalGroup(2, [30], 100)
    # Assign a mesh size to all the points:
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), Lc1)
    # gmsh.model.mesh.setSize(gmsh.model.getBoundary([31], False, False, True), Lc2)
    gmsh.model.mesh.setAlgorithm(2, 30, 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.msh" % model_name)
    gmsh.finalize()


def GenerateWheelMeshFileTreadVersion(outputFile, wheelDimensions, meshSize):
    """05/07/2021 new code to generate a simple wheel in a sectorial manner"""
    gmsh.initialize()
    gmsh.model.add(outputFile)

    mesh_size = meshSize  # meshSize # 0.7
    tread_angle = 20 * np.pi / 180

    cm = 1.
    center_x = 0 * cm
    center_y = 15 * cm
    # center_y = 15 * cm
    radius1, radius2 = wheelDimensions
    layers_number = 8

    factory = gmsh.model.occ
    factory.addPoint(center_x, center_y, 0, mesh_size, 1)
    factory.addPoint(-np.sin(tread_angle) * radius2 + center_x,
                     -np.cos(tread_angle) * radius2 + center_y, 0, mesh_size, 2)
    factory.addPoint(np.sin(tread_angle) * radius2 + center_x,
                     -np.cos(tread_angle) * radius2 + center_y, 0, mesh_size, 3)

    factory.addPoint(-np.sin(tread_angle) * radius1 + center_x,
                     -np.cos(tread_angle) * radius1 + center_y, 0, mesh_size, 4)
    factory.addPoint(np.sin(tread_angle) * radius1 + center_x,
                     -np.cos(tread_angle) * radius1 + center_y, 0, mesh_size, 5)

    factory.addCircleArc(2, 1, 3, 1)
    factory.addCircleArc(4, 1, 5, 2)

    factory.addLine(4, 2, 3)
    factory.addLine(3, 5, 4)

    factory.addCurveLoop([2, 3, 1, 4], 1)
    factory.addPlaneSurface([1], 1)

    out_list = []
    out = [(2, 1)]
    out_list.append(1)
    for t in range(0, layers_number):
        out = gmsh.model.occ.copy([(2, out[0][1])])
        gmsh.model.occ.rotate(out, center_x, center_y, 0, 0, 0, 1, 2 * tread_angle)
        out_list.append(out[0][1])

    factory.synchronize()

    # Assign a mesh size to all the points:
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.geo.remove([(1, 1)])
    #
    # # gmsh.model.mesh.setSize(gmsh.model.getBoundary([31], False, False, True), Lc2)
    # # gmsh.model.mesh.setAlgorithm(2, 30, 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.msh" % outputFile)

    # Launch the GUI to see the results:
    # import sys
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    gmsh.finalize()



def GenerateWheelMeshFileAsymVersion(outputFile, wheelDimensions, meshSize, refMesh):
    gmsh.initialize()
    gmsh.model.add(outputFile)

    mesh_size = 2.#meshSize # 0.7
    tread_angle = 20 * np.pi / 180

    cm = 1.
    center_x = 0 * cm
    center_y = 50 * cm
    # center_y = 15 * cm
    radius1, radius2 = wheelDimensions
    layers_number = 8

    factory = gmsh.model.occ
    factory.addPoint(center_x, center_y, 0, mesh_size, 1)
    factory.addPoint(-np.sin(tread_angle) * radius2 + center_x,
                     -np.cos(tread_angle) * radius2 + center_y, 0, mesh_size, 2)
    factory.addPoint(np.sin(tread_angle) * radius2 + center_x,
                     -np.cos(tread_angle) * radius2 + center_y, 0, mesh_size, 3)

    factory.addPoint(-np.sin(tread_angle) * radius1 + center_x,
                     -np.cos(tread_angle) * radius1 + center_y, 0, mesh_size, 4)
    factory.addPoint(np.sin(tread_angle) * radius1 + center_x,
                     -np.cos(tread_angle) * radius1 + center_y, 0, mesh_size, 5)

    factory.addCircleArc(2, 1, 3, 1)
    factory.addCircleArc(4, 1, 5, 2)

    factory.addLine(4, 2, 3)
    factory.addLine(3, 5, 4)

    factory.addCurveLoop([2, 3, 1, 4], 1)
    factory.addPlaneSurface([1], 1)

    out_list = []
    if refMesh is None:
        factory.addRectangle(-2, -10, 0, 4, 20, 0)
        # factory.addRectangle(-1, 0, 0, 2, 2, 0)
        gmsh.model.occ.cut([(2, 1)], [(2, 0)], 3)

        out2 = [(2, 3)]
        out_list.append(3)
        for t in range(0, layers_number):
            out2 = gmsh.model.occ.copy([(2, out2[0][1])])
            gmsh.model.occ.rotate(out2, center_x, center_y, 0, 0, 0, 1, 2 * tread_angle)
            out_list.append(out2[0][1])

    else:
        out = [(2, 1)]
        out_list.append(1)
        for t in range(0, layers_number):
            out = gmsh.model.occ.copy([(2, out[0][1])])
            gmsh.model.occ.rotate(out, center_x, center_y, 0, 0, 0, 1, 2 * tread_angle)
            out_list.append(out[0][1])

    factory.synchronize()

    # Assign a mesh size to all the points:
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.geo.remove([(1, 1)])
        #
        # # gmsh.model.mesh.setSize(gmsh.model.getBoundary([31], False, False, True), Lc2)
        # # gmsh.model.mesh.setAlgorithm(2, 30, 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.msh" % outputFile)

    gmsh.finalize()

    return 0


def MergeMeshes(outputFile, meshFiles):
    gmsh.initialize()
    model_name = outputFile
    gmsh.model.add(model_name)
    for mesh in meshFiles:
        gmsh.merge(mesh)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.msh" % outputFile)

    gmsh.finalize()


def GenerateCoincidentHFLFMeshes(romMeshFile, refMeshFile, interRadius, wheelDim, meshSize, version="Standard"):
    wheel_radius = interRadius
    if version=="Asym":
        GenerateWheelMeshFileAsymVersion(outputFile=romMeshFile, wheelDimensions=(wheel_radius, wheelDim[1]), meshSize=meshSize, refMesh=None)
        GenerateWheelMeshFileAsymVersion(outputFile="wheel_inter", wheelDimensions=(wheelDim[0], wheel_radius), meshSize=meshSize, refMesh=1)
    elif version=="Tread":
        GenerateWheelMeshFileTreadVersion(outputFile=romMeshFile, wheelDimensions=(wheel_radius, wheelDim[1]), meshSize=meshSize)
        GenerateWheelMeshFileTreadVersion(outputFile="wheel_inter", wheelDimensions=(wheelDim[0], wheel_radius), meshSize=meshSize)
    elif version=="Standard":
        GenerateWheelMeshFileStandardVersion(outputFile=romMeshFile, wheelDimensions=(wheel_radius, wheelDim[1]), meshSize=meshSize)
        GenerateWheelMeshFileStandardVersion(outputFile="wheel_inter", wheelDimensions=(wheelDim[0], wheel_radius), meshSize=meshSize)

    MergeMeshes(refMeshFile, [romMeshFile+".msh", "wheel_inter.msh"])

def CheckIntegrity():
    wheelDimensions=(8.,15.)
    GenerateCoincidentHFLFMeshes(romMeshFile="wheel_romStandard",refMeshFile="wheel_refStandard",interRadius=11.5,wheelDim=wheelDimensions,meshSize=1.0,version="Standard")
    GenerateCoincidentHFLFMeshes(romMeshFile="wheel_romTread",refMeshFile="wheel_refTread",interRadius=11.5,wheelDim=wheelDimensions,meshSize=1.0,version="Tread")
    wheelDimensions=(20.,50.)
    GenerateCoincidentHFLFMeshes(romMeshFile="wheel_romAsym",refMeshFile="wheel_refAsym",interRadius=39.0,wheelDim=wheelDimensions,meshSize=1.0,version="Asym")
    return "OK"

if __name__ =="__main__":
    CheckIntegrity()