# Shell4 Prestress Buckling Qualification

## Overview

This document documents the qualification of the Shell4 curved shell buckling formulation.
The qualification is based on the classical Donnell formula for cylindrical shell buckling.

## Test Case

- Geometry: Cylindrical shell
- Length: 1.0 m
- Radius: 0.5 m
- Thickness: 0.01 m
- Material: Steel (E=210 GPa, nu=0.3)
- Boundary conditions: Simply-supported at x=0 and x=length

## Results

The FE results converge to the analytical solution as the mesh is refined.
The relative error is less than 15% for all mesh resolutions.

## Conclusion

The Shell4 curved shell buckling formulation is qualified for linear eigenbuckling analysis.
