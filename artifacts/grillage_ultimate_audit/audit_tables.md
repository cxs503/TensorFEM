## Plate field: Navier series vs independent FD (T11)

| case | patch [m] | peak moment series [N m/m per Pa] | FD | error | p_ult module [MPa] | p_ult FD [MPa] | error | governing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| light-ice + ballast head | 0.35 x 0.35 | 0.0102348 | 0.0102329 | -0.0184 % | 3.3546 | 3.3552 | +0.0184 % | plate_local_bending |
| design-ice + design head | 0.70 x 0.35 | 0.0138788 | 0.0138773 | -0.0107 % | 2.3298 | 2.3300 | +0.0108 % | plate_local_bending |
| severe-ice + tank head | 1.05 x 0.35 | 0.0149363 | 0.0149357 | -0.0045 % | 1.9688 | 1.9689 | +0.0046 % | plate_local_bending |

FD grid convergence (severe-case patch), peak moment: n_x=60: 0.0720 %, n_x=120: 0.0180 %, n_x=240: 0.0045 %

## Member-governed combined cases (T12, T13)

| case | sigma_x [MPa] | patch [m] | p_ult [MPa] | U_plate | U_stiff | U_frame | governing | margin | closed form [MPa] | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| light ice + ballast head | 0.00 | 0.55 x 0.55 | 1.9149 | 0.838 | 1.0000 | 0.066 | longitudinal_stiffener | 1.915 | 1.9149 | -5.60e-06 % |
| design ice + design head | 53.25 | 1.10 x 0.55 | 1.2042 | 0.759 | 1.0000 | 0.093 | longitudinal_stiffener | 0.803 | 1.2042 | -4.22e-07 % |
| severe ice + tank head | 106.50 | 1.65 x 0.55 | 0.9627 | 0.718 | 1.0000 | 0.119 | longitudinal_stiffener | 0.481 | 0.9627 | -1.04e-05 % |

Cross-method (T13): elastic member limit 1.2042 MPa vs rigid-plastic grillage 5.7211 MPa (ratio 4.751, plastic system transverse_frame).

Governing-branch census: case study (t=20 mm): {'plate_local_bending': 6}, member-governed (t=25 mm): {'longitudinal_stiffener': 5, 'plate_local_bending': 1}, slender (t=13 mm): {'plate_local_bending': 6}
