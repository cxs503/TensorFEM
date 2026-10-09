# Advanced contact and mixed-mode cohesive validation

## Scope

`advanced_contact.py` assembles small-sliding, frictionless node-to-segment
constraints. It uses reference normals and projection weights and deliberately
does not claim finite-sliding search, friction, mortar contact, or Hertz FE.
The existing Hertz routine remains labelled an analytical reference only.

The multi-contact benchmark has two slave nodes and two deformable master
segments. Its reactions are independently obtained from the Schur complement

`lambda = -(C K^-1 C^T)^-1 (g0 + C K^-1 f)`.

The production KKT result is checked for reaction error below 3%, equilibrium,
zero active gaps and complementarity. Rigid translation invariance verifies
the assembled interpolation weights.

## Mixed-mode cohesive law

`mixed_mode_cohesive.py` implements monotonic proportional paths with the
Benzeggagh--Kenane criterion

`Gc = GIc + (GIIc-GIc) (Gs/(Gn+Gs))^eta`.

Pure mode I, pure mode II and a 3:4 mixed path integrate the complete bilinear
traction-separation curve. Each numerical work integral must differ from its
closed-form BK energy by strictly less than 3%; the current 20,001-point
trapezoidal integrations are effectively exact. An equal modal-work case also
checks the BK expression directly.

Reference: Benzeggagh, M. L. and Kenane, M. (1996), *Measurement of
mixed-mode delamination fracture toughness of unidirectional glass/epoxy
composites with mixed-mode bending apparatus*, Composites Science and
Technology 56(4), 439-449, DOI 10.1016/0266-3538(96)00005-X.

This constitutive reference does not claim arbitrary non-proportional loading,
crack insertion, branching or a general fracture solver.
