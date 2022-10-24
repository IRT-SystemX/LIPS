#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def PlotTagsInMesh(mesh,refNumByRegion,tagsToPlot="ALL",plotPoints=True):
    boundary={tagName:mesh.pts(PIDs=mesh.pid_in_regions(tagValue)) for tagName,tagValue in refNumByRegion.items()}

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if plotPoints:
        pts=mesh.pts()
        xs, ys, zs=pts[0,:],pts[1,:],pts[2,:]
        ax.scatter(xs, ys, zs)

    if tagsToPlot=="ALL":
        tagsRepresented=refNumByRegion.keys()
    elif isinstance(tagsToPlot, str):
        tagsRepresented=[tagsToPlot]
    else:
        try:
            tagsRepresented = (elem for elem in tagsToPlot)
        except TypeError:
            raise Exception(tagsToPlot, 'is not iterable')

    for tagName in tagsRepresented:
        try:
            ptsInMyTag=boundary[tagName]
        except KeyError:
            raise Exception(tagName+"does not exist within refNumByRegion provided")
        xs, ys, zs=ptsInMyTag[0,:],ptsInMyTag[1,:],ptsInMyTag[2,:]
        ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


import PhysicalTools.PhysicalSolver.GetfemBricks.MeshTools as gfMesh

if __name__ =="__main__":
    refNumByRegion = {"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3}
    myMesh=gfMesh.Generate3DWheelMesh(wheelDimensions=(8,15,2.5),meshSize=1.0,refNumByRegion=refNumByRegion)
    PlotTagsInMesh(mesh=myMesh,refNumByRegion=refNumByRegion,tagsToPlot="CONTACT_BOUND")
