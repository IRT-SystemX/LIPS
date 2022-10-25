#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string
import random

import numpy as np

import getfem as gf
gf.util_trace_level(1)

def GetNumberOfNodes(mesh):
    return mesh.nbpts()

def GetNumberOfElements(mesh):
    return mesh.nbcvs()

def GetNodesInRegion(mesh,regionId):
    pts=mesh.pts().transpose()
    ptsIdInRegion=mesh.pid_in_regions(regionId)
    return pts[ptsIdInRegion]

def InterpolateFieldOnMesh(model,fieldExpression,mesh):
    pts=mesh.pts()
    field=model.interpolation(fieldExpression,pts,mesh)
    return field

def ExportMeshInVtk(mesh,meshFile):
    mesh.export_to_vtk(meshFile)

def ExportMeshInMsh(mesh,meshFile):
    mesh.export_to_pos(meshFile)

def ImportGmshMesh(meshFile):
    return gf.Mesh('import', 'gmsh', meshFile)

def Generate2DBeamMesh(meshSize,RefNumByRegion):
    NX = meshSize 
    mesh = gf.Mesh('regular simplices', np.arange(0,40+10/NX,10/NX), np.arange(0,10+10/NX,10/NX))

    #Boundaries definition
    P=mesh.pts()
    ctop=(abs(P[1,:]-10) < 1e-6)
    cbot=(abs(P[1,:]) < 1e-6)
    cright=(abs(P[0,:]-40) < 1e-6)
    cleft=(abs(P[0,:]) < 1e-6)

    zoneByRegion={
               "Left" : (RefNumByRegion["Left"],cleft),
               "Right" : (RefNumByRegion["Right"],cright),
               "Bottom" : (RefNumByRegion["Bottom"],cbot),
               "Top" : (RefNumByRegion["Top"],ctop),
            }

    for _,region in zoneByRegion.items():
        regionId,zone=region
        pidPoints=gf.compress(zone, range(0, mesh.nbpts()))
        facesInPid=mesh.faces_from_pid(pidPoints)
        mesh.set_region(regionId,facesInPid)
    return mesh

def GenerateSimpleMesh(meshSize):
    mesh=gf.Mesh('regular simplices', np.arange(0,4+1/meshSize,1/meshSize), np.arange(0,1+1/meshSize,1/meshSize))

    #Boundaries definition
    P=mesh.pts()
    ctop=(abs(P[1,:]-1) < 1e-6)
    cbot=(abs(P[1,:]) < 1e-6)
    cright=(abs(P[0,:]-4) < 1e-6)
    cleft=(abs(P[0,:]) < 1e-6)

    refNumByRegion={
               "TOP" : 1,
               "LEFT" : 2,
               "BOTTOM" : 3,
               "RIGHT" : 4
            }

    zoneByRegion={
               "LEFT" : (refNumByRegion["LEFT"],cleft),
               "RIGHT" : (refNumByRegion["RIGHT"],cright),
               "BOTTOM" : (refNumByRegion["BOTTOM"],cbot),
               "TOP" : (refNumByRegion["TOP"],ctop),
            }

    for _,region in zoneByRegion.items():
        regionId,zone=region
        pidPoints=gf.compress(zone, range(0, mesh.nbpts()))
        facesInPid=mesh.faces_from_pid(pidPoints)
        mesh.set_region(regionId,facesInPid)
    return mesh,refNumByRegion

def GenerateTagged3DBeam(meshSize):
    mesh=gf.Mesh('regular simplices', np.arange(0,4+1/meshSize,1/meshSize), np.arange(0,1+1/meshSize,1/meshSize),np.arange(0,2+1/meshSize,1/meshSize))

    #Boundaries definition
    P=mesh.pts()
    cxMax=(abs(P[0,:]-4) < 1e-6)
    cxMin=(abs(P[0,:]) < 1e-6)
    cyMax=(abs(P[1,:]-1) < 1e-6)
    cyMin=(abs(P[1,:]) < 1e-6)
    czMax=(abs(P[2,:]-2) < 1e-6)
    czMin=(abs(P[2,:]) < 1e-6)

    zoneByRegion={
               "XMAX" : (1,cxMax),
               "XMIN" : (2,cxMin),
               "YMAX" : (3,cyMax),
               "YMIN" : (4,cyMin),
               "ZMAX" : (5,czMax),
               "ZMIN" : (6,czMin)
            }

    for _,region in zoneByRegion.items():
        regionId,zone=region
        pidPoints=gf.compress(zone, range(0, mesh.nbpts()))
        facesInPid=mesh.faces_from_pid(pidPoints)
        mesh.set_region(regionId,facesInPid)
    
    refNumByRegion={regionName:regionVal[0] for regionName,regionVal in zoneByRegion.items()}
    return mesh,refNumByRegion

def GenerateWheelMesh(wheelDimensions,meshSize,RefNumByRegion):
    r_inner, r_ext = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    if r_inner >= r_ext:
        raise Exception("r_inner should be lower than r_ext")
    mo1 = gf.MesherObject('ball', [0., center], r_ext)
    mo2 = gf.MesherObject('ball', [0., center], r_inner)
    mo3 = gf.MesherObject('set minus', mo1, mo2)

    mesh = gf.Mesh('generate', mo3, meshSize, 2)
    mesh=AddWheelBoundaryConditions(mesh,wheelDimensions,RefNumByRegion)
    return mesh

def AddWheelBoundaryConditions(mesh, wheelDimensions, RefNumByRegion):
    r_inner, r_ext = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    eps = 0.1
    # fb1 = mesh.outer_faces_in_box([-r_inner - eps, center - r_inner - eps], [r_inner + eps, center + r_inner + eps])  # Boundary of the hole
    # fb2 = mesh.outer_faces_with_direction([0., -1.], np.pi/2) # Contact boundary of the wheel
    fb2 = mesh.outer_faces_in_box([-center - eps, 0.-eps], [center+eps, center+eps]) # Contact boundary of the wheel
    # fb3 = mesh.outer_faces_in_box([-r_ext - eps, - eps], [r_ext + eps, center + r_ext + eps]) # Exterior boundary of tire

    fb1 = mesh.outer_faces_in_ball([0., center], r_inner + eps)
    fb3 = mesh.outer_faces_in_ball([0., center], r_ext + eps)
    fb4 = mesh.outer_faces_with_direction([0., 1.], np.pi/2)

    fb4 = mesh.outer_faces_with_direction([-np.cos(-np.pi/3), -np.sin(-np.pi/3)], 0.0)
    fb5 = mesh.outer_faces_with_direction([np.cos(np.pi/3), np.sin(np.pi/3)], 0.)

    mesh.set_region(RefNumByRegion["HOLE_BOUND"], fb1)
    mesh.set_region(RefNumByRegion["CONTACT_BOUND"], fb2)
    mesh.region_subtract(RefNumByRegion["CONTACT_BOUND"], RefNumByRegion["HOLE_BOUND"])
    mesh.set_region(RefNumByRegion["EXTERIOR_BOUND"], fb3)
    mesh.region_subtract(RefNumByRegion["EXTERIOR_BOUND"], RefNumByRegion["HOLE_BOUND"])

    return mesh

def Generate3DWheelMesh(wheelDimensions,meshSize,refNumByRegion):
    r_inner, r_ext, thickness = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    if r_inner >= r_ext:
        raise Exception("r_inner should be lower than r_ext")
    mo1 = gf.MesherObject('cylinder', [0., center, 0.],[0., 0., 1.], thickness,r_ext)
    mo2 = gf.MesherObject('cylinder', [0., center, 0.],[0., 0., 1.], thickness, r_inner)
    mo3 = gf.MesherObject('set minus', mo1, mo2)

    mesh = gf.Mesh('generate', mo3, meshSize, 2)
    Add3DWheelBoundaryConditions(mesh,wheelDimensions,refNumByRegion)
    return mesh

def Add3DWheelBoundaryConditions(mesh, wheelDimensions, RefNumByRegion):
    r_inner, r_ext, thickness = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    eps = 0.1
    fb1 = mesh.outer_faces_in_ball([0., center, thickness/2], max(r_inner,thickness) + eps)
    fb2 = mesh.outer_faces_in_box([-center - eps, 0.-eps, -eps], [center+eps, center+eps, thickness+eps]) # Contact boundary of the wheel
    fb3 = mesh.outer_faces_in_ball([0., center, thickness/2], max(r_ext,thickness) + eps)

    mesh.set_region(RefNumByRegion["HOLE_BOUND"], fb1)
    mesh.set_region(RefNumByRegion["CONTACT_BOUND"], fb2)
    mesh.region_subtract(RefNumByRegion["CONTACT_BOUND"], RefNumByRegion["HOLE_BOUND"])
    mesh.set_region(RefNumByRegion["EXTERIOR_BOUND"], fb3)
    mesh.region_subtract(RefNumByRegion["EXTERIOR_BOUND"], RefNumByRegion["HOLE_BOUND"])

    return mesh

def TagWheelMesh(mesh,wheelDimensions,center,refNumByRegion):
    epsilon = 0.1
    origin_x,origin_y=center
    r_inner,r_ext=wheelDimensions
    all_faces = mesh.outer_faces_in_box([-origin_x - r_ext - epsilon, -origin_y - r_ext -epsilon], [origin_x + r_ext + epsilon, origin_y + r_ext +epsilon])
    rim_faces = mesh.outer_faces_in_ball([origin_x,origin_y], r_inner + epsilon)

    mesh.set_region(refNumByRegion["HOLE_BOUND"], rim_faces)
    mesh.set_region(refNumByRegion["CONTACT_BOUND"], all_faces)
    mesh.region_subtract(refNumByRegion["CONTACT_BOUND"], refNumByRegion["HOLE_BOUND"])

    return mesh

def GenerateWheelMeshRolling(wheelDimensions,meshSize,RefNumByRegion):
    r_inner, r_ext = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    if r_inner >= r_ext:
        raise Exception("r_inner should be lower than r_ext")
    mo1 = gf.MesherObject('ball', [0., center], r_ext)
    mo2 = gf.MesherObject('ball', [0., center], r_inner)
    mo3 = gf.MesherObject('set minus', mo1, mo2)

    mesh = gf.Mesh('generate', mo3, meshSize, 2)
    mesh=AddWheelBoundaryConditionsRolling(mesh,wheelDimensions,RefNumByRegion)
    return mesh

def AddWheelBoundaryConditionsRolling(mesh,wheelDimensions,RefNumByRegion):
    r_inner, r_ext = wheelDimensions
    center = 15 # Impose center used to be equal to r_ext
    eps = 0.1
    # fb1 = mesh.outer_faces_in_box([-r_inner - eps, center - r_inner - eps], [r_inner + eps, center + r_inner + eps])  # Boundary of the hole
    fb2 = mesh.outer_faces_with_direction([0., -1.], np.pi/2) # Contact boundary of the wheel
    # fb3 = mesh.outer_faces_in_box([-r_ext - eps, - eps], [r_ext + eps, center + r_ext + eps]) # Exterior boundary of tire

    fb1 = mesh.outer_faces_in_ball([0., center], r_inner + eps)
    fb3 = mesh.outer_faces_in_ball([0., center], r_ext + eps)
    fb4 = mesh.outer_faces_with_direction([0., 1.], np.pi/2)

    mesh.set_region(RefNumByRegion["HOLE_BOUND"], fb1)
    mesh.set_region(RefNumByRegion["CONTACT_BOUND"], fb3)
    mesh.region_subtract(RefNumByRegion["CONTACT_BOUND"], RefNumByRegion["HOLE_BOUND"])
    mesh.set_region(RefNumByRegion["EXTERIOR_BOUND"], fb3)
    mesh.region_subtract(RefNumByRegion["EXTERIOR_BOUND"], RefNumByRegion["HOLE_BOUND"])
    if "NEUMANN_BOUND" in RefNumByRegion:
        mesh.set_region(RefNumByRegion["NEUMANN_BOUND"], fb4)

    return mesh

def ImportGMSHWheel(meshFile,wheelDimensions,RefNumByRegion):
    mesh=ImportGmshMesh(meshFile)
    return AddWheelBoundaryConditions(mesh, wheelDimensions, RefNumByRegion)
