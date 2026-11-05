# Shell4 Curved Geometric Stiffness

## Overview

This document describes the geometric stiffness formulation for the Shell4 curved shell element.
The geometric stiffness matrix is used in initial-stress eigenbuckling analysis to account for
the effect of membrane stresses on the bending stiffness of the shell.

## Formulation

The geometric stiffness matrix for a cylindrical shell patch is derived from the strain energy
due to the initial stress. For a cylindrical shell under axial compression, the geometric strain
is given by:

$$\epsilon_x^g = \frac{1}{2} \left(\frac{\partial w}{\partial x}\right)^2$$

where $w$ is the transverse displacement. The geometric stiffness matrix is:

$$K_g = \int N_x \frac{\partial w}{\partial x} \frac{\partial w}{\partial x} dA$$

where $N_x$ is the axial force resultant per unit length.

## Implementation

The geometric stiffness matrix is integrated using 2x2 Gauss quadrature over the natural
coordinates of the Q4 element. The transverse displacement DOFs are used to construct the
geometric strain-displacement matrix.

## Verification

The formulation is verified against the classical Donnell formula for cylindrical shell buckling.
The FE results converge to the analytical solution as the mesh is refined.

## References

- Donnell, L. H. (1934). "A new theory for the buckling of thin cylinders under axial compression and bending." Transactions of the ASME, 56, 795-806.
