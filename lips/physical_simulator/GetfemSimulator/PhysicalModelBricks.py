#!/usr/bin/env python
# -*- coding: utf-8 -*-
import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver

behaviourLawByName={"LinearElasticity":PhySolver.AddLinearElasticity,
                    "IncompressibleMooneyRivlin":PhySolver.AddIncompMooneyRivlin,
                    "SaintVenantKirchhoff":PhySolver.AddSaintVenantKirchhoff,
                    "CompressibleNeoHookean":PhySolver.AddCompressibleNeoHookean
                    }

sourceTypeByName={"Uniform":PhySolver.AddUniformSourceTerm,"Variable":PhySolver.AddVariableSourceTerm}

dirichletByName={"scalar": PhySolver.AddDirichletCondition,
                "vector": PhySolver.AddDirichletConditionVector,
                "GlobalVector": PhySolver.AddDirichletConditionWithSimplification,
                "rhs": PhySolver.AddDirichletConditionRHS,
                "AnglePiloted": PhySolver.AddRollingCondition
                }

neumannByName={"RimRigidityNeumann":PhySolver.AddRimRigidityNeumannCondition,
                       "StandardNeumann":PhySolver.AddNeumannCondition}

contactTypeByName={
               "NoFriction":PhySolver.AddUnilatContact,
               "Inclined":PhySolver.AddInclinedUnilatContactWithFric,
               "Plane":PhySolver.AddUnilatContactWithFric,
               "PlanePenalized":PhySolver.AddPenalizedUnilatContactWithFric
               }

rollingTypeByName = {"AnglePiloted": PhySolver.AddRollingCondition,
                    "DIS_Rolling": PhySolver.AddDisplacementImposedRollingCondition,
                    "FORC_Rolling": PhySolver.AddForceImposedRollingCondition 
                    }
