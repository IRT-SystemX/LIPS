#!/usr/bin/env python
# -*- coding: utf-8 -*-

import getfem as gf
gf.util_trace_level(1)

#Integration Methods
def DefineIntegrationMethodsByOrder(mesh, order):
    return gf.MeshIm(mesh, order)


def DefineCompositeIntegrationMethodsByName(mesh,name):
    return gf.MeshIm(mesh, gf.Integ(name))