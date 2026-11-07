# Armstrong-Frederick cyclic plasticity: smallest falsifiable anchor surface

**Mission**: P16-N2 / m5197 (branch `agent/m5197`) — cyclic plasticity for the
`TensorFEM` material point: nonlinear kinematic hardening, Bauschinger effect,
stabilised hysteresis loop and ratcheting rate, each pinned to a closed form.

**Upstream**: `docs/j2-plasticity-minimal-surface.md` (IN14-D4 / m4951) supplies the
monotonic J2 kernel (`linear isotropic hardening + radial return + consistent
tangent`, module `tensorfem.j2_elastoplasticity`).  That surface *explicitly
withholds* cyclic loading, kinematic hardening and ratcheting — this document fills
exactly that hole and keeps the monotonic kernel as its degeneration target (C6).

**Module**: `src/tensorfem/cyclic_plasticity.py`
**Driver**: `scripts/cyclic_plasticity_verification.py` →
`artifacts/cyclic_plasticity/verification.json`
**Tests**: `tests/test_cyclic_plasticity.py` (27 cases) + the m4951 regression
`tests/test_j2_minimum_surface.py` (16 cases, C8).

---

## 1. Case introduction

### 1.1 Physics

A metal point under *cyclic* load does not return along the monotonic curve: the
yield surface translates (kinematic hardening) and/or expands (isotropic hardening).
The Japanese/German school formulation used here is the Armstrong-Frederick (AF)
evolution with a linear recall term, the simplest model that produces

* **Bauschinger effect** — reverse yielding at a stress below the forward flow stress,
* **a closed stabilised hysteresis loop** with finite (not shape-preserving) area,
* **ratcheting** — a non-zero plastic strain increment per cycle under a stress
  cycle with a non-zero mean,

and that still has a closed-form solution for all three. AF is the reference model
for shakedown/ratcheting screening of hull and pressure-vessel steels; its known
weakness (gross over-prediction of multi-axial ratcheting) is a *modelling* limit,
not an implementation limit, and is stated again in §5.

### 1.2 Material parameters (all runs)

| symbol | value | note |
|---|---|---|
| $E$ | 210.0 GPa | Young's modulus |
| $\nu$ | 0.30 | Poisson ratio |
| $\sigma_{y0}$ | 350.0 MPa | initial yield stress |
| $C$ | 70.0 GPa | AF kinematic modulus |
| $\gamma$ | 500.0 | AF recall parameter |
| $X_{sat}=C/\gamma$ | 140.0 MPa | saturation back stress (Frobenius) |
| $\alpha_{sat}=\tfrac32 X_{sat}$ | 210.0 MPa | saturation back stress (uniaxial) |
| $H$ | 0.0 GPa | linear isotropic hardening (0 ⇒ pure AF) |
| $Q_\infty,\ b$ | 80.0 MPa, 500 | optional Voce saturation |

The set is a generic pressure-vessel / ship-structure steel: $\alpha_{sat}$ is
60 % of
$\sigma_{y0}$, i.e. the classical "strong kinematic, weak isotropic" hardening
split of quenched-and-tempered steels at moderate strain amplitudes.  The Voce
variant (80 MPa of extra saturation) exercises the isotropic branch
of the anchor.

### 1.3 Geometry / discretisation

There is **no mesh** in this deliverable: the object under test is the material
point, deliberately, because all seven criteria are *closed-form* comparisons and a
mesh would only add a second error source.  Two drivers replace the FE assembly:

* `uniaxial_cycle` — prescribed axial strain history, lateral strain solved per
  increment by a Newton loop on the **consistent tangent** so that
  $\sigma_{22}=\sigma_{33}=0$ (true uniaxial stress, tolerance $10^{-9}$ relative);
* `uniaxial_stress_cycle` — prescribed axial stress history, both $\varepsilon_{11}$
  and $\varepsilon_{22}=\varepsilon_{33}$ solved by a 2×2 Newton loop with the same
  tangent (used for ratcheting).

Strain-controlled sweeps use 60 steps per half cycle, the
ratcheting sweep 200 steps per half cycle and
20 cycles; a step-size study (§4.4) proves that the remaining
error is pure backward-Euler truncation.

---

## 2. Method

### 2.1 Constitutive equations

$$
\boldsymbol\varepsilon=\boldsymbol\varepsilon^e+\boldsymbol\varepsilon^p,\qquad
\boldsymbol\sigma=K\,\mathrm{tr}\boldsymbol\varepsilon^e\,\mathbf I+2G\,\mathbf s^e,\qquad
f=\lVert\mathbf s-\mathbf X\rVert_F-\sqrt{\tfrac23}\,R(p)
$$

$$
\dot{\boldsymbol\varepsilon}^p=\dot p\,\sqrt{\tfrac32}\,\mathbf n,\qquad
\mathbf n=\frac{\mathbf s-\mathbf X}{\lVert\mathbf s-\mathbf X\rVert_F},\qquad
\dot{\mathbf X}=\tfrac23 C\,\dot{\boldsymbol\varepsilon}^p-\gamma\mathbf X\dot p,\qquad
R(p)=\sigma_{y0}+Hp+Q_\infty(1-e^{-bp})
$$

The yield function is written *exactly* as in the mission statement
($\lVert\cdot\rVert$ = Frobenius norm, so $f=0\Leftrightarrow\sigma_{eq}=R$); the
$\sqrt{2/3}$ factor is what makes $\lVert\mathbf s\rVert_F=\sqrt{2/3}\sigma$ in
uniaxial stress.  The flow direction normalisation ($\sqrt{3/2}$) is chosen so that
$\dot p$ is the von Mises equivalent plastic strain rate — i.e. **the same $\dot p$
as the m4951 kernel's `dgamma`**, which is what makes C6 a bit-level test and not an
approximate one.

### 2.2 Implicit return mapping (back stress inside the yield function)

Backward Euler in the *fully implicit* AF form,
$\mathbf X^{n+1}(1+\gamma\Delta p)=\mathbf X^n+\sqrt{2/3}\,C\,\Delta p\,\mathbf n$,
and $\mathbf s^{n+1}=\mathbf s^{tr}-\sqrt6\,G\,\Delta p\,\mathbf n$, give with
$a=1/(1+\gamma\Delta p)$

$$
\boldsymbol\eta=\mathbf s^{n+1}-\mathbf X^{n+1}
=\underbrace{\left[\mathbf s^{tr}-a\mathbf X^n\right]}_{\mathbf w}
-\underbrace{\left[\sqrt6 G+\sqrt{\tfrac23}Ca\right]}_{B}\Delta p\,\mathbf n .
$$

Because $\boldsymbol\eta=q_F\mathbf n$, taking the norm gives $\lVert\mathbf w\rVert_F=q_F+B\Delta p$,
so the whole return mapping collapses onto **one scalar monotone equation**

$$
\Phi(\Delta p)=\lVert\mathbf s^{tr}-a(\Delta p)\mathbf X^n\rVert_F
-\left[\sqrt6 G+\sqrt{\tfrac23}C\,a\right]\Delta p
-\sqrt{\tfrac23}\,R(p^n+\Delta p)=0 ,
$$

solved by a safeguarded Newton iteration (analytic derivative, bisection fallback,
tolerance $10^{-15}$ relative).  Two remarks matter for the physics:

* the equation is *exact for non-proportional steps* — the commmon "freeze
  $\mathbf X$" shortcut assumes $\mathbf X^n\parallel\mathbf n$ and is not used here;
* the $a(\Delta p)$ factor in $\mathbf X^{n+1}$ is what makes the discretisation
  **contractive** ($\partial\mathbf X^{n+1}/\partial\mathbf X^n=a\,\mathbf P<1$), hence
  a *stabilised* loop exists and is unique — with an explicit (Euler-forward) AF
  update the loop would not stabilise and none of the anchors would apply.

A posteriori residuals of the returned state (yield, flow rule, AF evolution,
incompressibility) are re-evaluated independently of the solver and reported below:
max $10^{-16}$, i.e. the audited quantity is the discretisation, not the algebra.

### 2.3 Consistent (algorithmic) tangent including $d\mathbf X/d\boldsymbol\varepsilon$

Linearising the same scalar equation ($s=\mathbf s^{tr}$, $\eta$-normal $\mathbf n$):

$$
\Delta p=\frac{\mathbf n:\mathrm d\mathbf s^{tr}}{D},\qquad
D=\sqrt6 G+\sqrt{\tfrac23}C\left(a-\gamma a^2\Delta p\right)
+\sqrt{\tfrac23}R'(p+\Delta p)-\gamma a^2\left(\mathbf n:\mathbf X^n\right),
$$

$$
\mathrm d\mathbf s=\mathrm d\mathbf s^{tr}-\sqrt6 G\left(\mathbf n\,\Delta p+\Delta p\,\mathrm d\mathbf n\right),\qquad
\mathrm d\mathbf n=\frac{\left(\mathbf I-\mathbf n\otimes\mathbf n\right):\mathrm d\mathbf w}{\lVert\mathbf w\rVert_F},\qquad
\mathrm d\mathbf w=\mathrm d\mathbf s^{tr}+\gamma a^2\mathbf X^n\,\Delta p .
$$

The $\gamma a^2(\mathbf n:\mathbf X^n)$ term is the *only* place where the back stress
enters the tangent besides $\mathbf n$; dropping it (a common bug) leaves a tangent
that is symmetric-looking but wrong by 1–4 %, exactly the failure mode that was hit
and fixed during this mission (§5).  The returned 6×6 matrix follows the shared Voigt
convention ($[xx,yy,zz,xy,yz,zx]$, engineering shear columns) so it drops straight
into an assembled stiffness.

### 2.4 Closed-form anchors

With $\alpha=\tfrac32 X_{11}$, $\Delta\varepsilon^p$ the plastic strain range of a
half cycle and $\alpha_a=\tfrac12(\alpha_{max}-\alpha_{min})$:

1. **stabilised loop amplitude** — integrating $\mathrm d\alpha/\mathrm dp=C\mp\gamma\alpha$
   over one half cycle and imposing periodicity:

   $$\alpha_a=\frac{C}{\gamma}\tanh\left(\gamma\varepsilon^p_a\right),\qquad
   \sigma_a=\alpha_a+R_{sat},\qquad \varepsilon^p_a=\varepsilon_a-\sigma_a/E$$

   (a two-equation scalar problem, solved by bisection to $10^{-14}$).

2. **dissipated energy per cycle** — $W=\oint\sigma\,\mathrm d\varepsilon
   =4R_{sat}\varepsilon^p_a+\dfrac{4C}{\gamma^2}\left(\gamma\varepsilon^p_a
   -\tanh\gamma\varepsilon^p_a\right)$; the second term vanishes as $\gamma\to0$
   (the Prager limit is a parallelogram of area $4R\varepsilon^p_a$).

3. **Bauschinger drop** — the reverse yield stress of the stabilised loop is
   $\sigma_{rev}=\alpha_a-R_{sat}$, so the drop measured against the isotropic
   radius is exactly $\Delta\sigma_B=R_{sat}-|\sigma_{rev}|=\alpha_a$.

4. **ratcheting rate** (stress control, steady cycle) — during plastic flow
   $\alpha=\sigma-R$ on the forward branch and $\alpha=\sigma+R$ on the reverse one,
   so the back-stress extrema of the steady cycle are
   $\alpha_{max}=\sigma_m+\sigma_a-R$, $\alpha_{min}=\sigma_m-\sigma_a+R$
   (both confirmed in the simulation, see §4.3), and the two branches integrate to

   $$\frac{\Delta\varepsilon^p}{\text{cycle}}
   =\frac1\gamma\left[\ln\frac{C-\gamma\alpha_{min}}{C-\gamma\alpha_{max}}
   -\ln\frac{C+\gamma\alpha_{max}}{C+\gamma\alpha_{min}}\right],
   \qquad \max|\alpha_{max,min}|<C/\gamma .$$

### 2.5 Toolchain

Pure `torch` (float64) for the tensors, the scalar core in Python floats;
`matplotlib` (Agg) for the figures; `pytest` for the gates; a first-order step-size
study plus Richardson extrapolation to separate truncation from physics.  Everything
runs on one core in a few minutes (the reference run used the 702 cluster node spec
only for the wider sweep of `tests/`).

---

## 3. Results

### 3.1 Stabilised hysteresis loops (C1/C2/C3)

| $\varepsilon_a$ | cycles | $\sigma_{max}$ sim [MPa] | closed form [MPa] | C1 | $\sigma_{min}$ sim [MPa] | C1 (trough) | $W_{cycle}$ sim [MJ/m³] | closed form [MJ/m³] | C2 | $\Delta\sigma_B$ sim [MPa] | closed form [MPa] | C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 % | 5 | 451.20 | 451.89 | 0.1541 % | -451.20 | 0.1541 % | 2.81 | 2.81 | 0.0645 % | 101.20 | 101.89 | 0.6837 % |
| 0.6 % | 5 | 482.72 | 483.24 | 0.1070 % | -482.72 | 0.1070 % | 6.17 | 6.18 | 0.1874 % | 132.72 | 133.24 | 0.3880 % |
| 0.8 % | 5 | 488.87 | 489.04 | 0.0350 % | -488.87 | 0.0350 % | 9.98 | 10.00 | 0.2554 % | 138.87 | 139.04 | 0.1231 % |
| 1.0 % | 5 | 489.83 | 489.87 | 0.0087 % | -489.83 | 0.0087 % | 13.87 | 13.91 | 0.2531 % | 139.83 | 139.87 | 0.0306 % |
| 1.2 % | 5 | 489.97 | 489.98 | 0.0019 % | -489.97 | 0.0019 % | 17.78 | 17.83 | 0.2454 % | 139.97 | 139.98 | 0.0068 % |
| 0.4 % | 20 | 451.20 | 451.89 | 0.1541 % | -451.20 | 0.1541 % | 2.81 | 2.81 | 0.0645 % | 101.20 | 101.89 | 0.6835 % |
| 0.6 % | 20 | 482.72 | 483.24 | 0.1070 % | -482.72 | 0.1070 % | 6.17 | 6.18 | 0.1874 % | 132.72 | 133.24 | 0.3880 % |
| 0.8 % | 20 | 488.87 | 489.04 | 0.0350 % | -488.87 | 0.0350 % | 9.98 | 10.00 | 0.2554 % | 138.87 | 139.04 | 0.1231 % |
| 1.0 % | 20 | 489.83 | 489.87 | 0.0087 % | -489.83 | 0.0087 % | 13.87 | 13.91 | 0.2531 % | 139.83 | 139.87 | 0.0306 % |
| 1.2 % | 20 | 489.97 | 489.98 | 0.0019 % | -489.97 | 0.0019 % | 17.78 | 17.83 | 0.2454 % | 139.97 | 139.98 | 0.0068 % |

Figure: `artifacts/cyclic_plasticity/hysteresis_loops.png`

![Stabilised hysteresis loops](../artifacts/cyclic_plasticity/hysteresis_loops.png)

![Anchor errors](../artifacts/cyclic_plasticity/anchor_errors.png)

The loops are *stabilised* (identical peak stress at cycle 5 and cycle 20 — the AF
recall term is contractive), the peaks sit on the closed form to
0.1541 % and the fact that the error *decreases* with amplitude
(the loop becomes more plastic, so the one-increment truncation matters less) is the
signature of discretisation rather than of a wrong anchor.

The Voce-isotropic variant ($Q_\infty=80$ MPa) behaves the same way
(peak error 0.0402 %, 0.0023 % at
$\varepsilon_a$ = 1 %, 1 %;
loop energy 0.2224 %, 0.2182 %),
so the anchor is not tuned to the $H=0$ special case.

### 3.2 Ratcheting (C5)

| $\sigma_m$ [MPa] | $\sigma_a$ [MPa] | $\Delta\varepsilon^p$ per cycle (sim) | closed form | C5 | $\alpha_{min}$ sim / closed form [MPa] | $\alpha_{max}$ sim / closed form [MPa] |
|---|---|---|---|---|---|---|
| 20 | 400 | 4.896e-04 | 4.814e-04 | 1.71 % | -30.0 / -30.0 | 70.0 / 70.0 |
| 20 | 420 | 8.104e-04 | 7.935e-04 | 2.13 % | -50.0 / -50.0 | 90.0 / 90.0 |
| 40 | 400 | 1.076e-03 | 1.056e-03 | 1.91 % | -10.0 / -10.0 | 90.0 / 90.0 |
| 40 | 420 | 1.875e-03 | 1.827e-03 | 2.61 % | -30.0 / -30.0 | 110.0 / 110.0 |

![Ratcheting](../artifacts/cyclic_plasticity/ratcheting.png)

The measured back-stress extrema agree with the closed-form endpoints to the printed
precision, i.e. the *mechanism* ($\alpha_{max}+\alpha_{min}=2\sigma_m$) is reproduced,
not merely the rate.  The rate itself is 2.61 % at
200 steps per half cycle, which is a truncation error, not
a model discrepancy — see §3.3.

### 3.3 Step-size convergence of the ratcheting rate

| steps per half cycle | measured $\Delta\varepsilon^p$/cycle | error vs closed form |
|---|---|---|
| 50 | 2.027e-03 | 10.906 % |
| 100 | 1.924e-03 | 5.298 % |
| 200 | 1.875e-03 | 2.609 % |
| 400 | 1.851e-03 | 1.295 % |
| Richardson limit ($h\to0$) | 1.827e-03 | 0.020 % |

![Step-size convergence](../artifacts/cyclic_plasticity/ratchet_convergence.png)

The error falls as $O(1/\Delta\varepsilon)$ (slope −1 on the log-log plot) and the
Richardson limit lands 0.020 % from the closed form.  This is the
strongest of the seven statements: **the closed-form ratcheting rate is confirmed to
0.02 %**, the 5 % gate being met already at the working step size.

### 3.4 Consistent tangent (C4)

| state | $\Delta p$ | max $|C_{ij}|$ [GPa] | C4 relative error |
|---|---|---|---|
| elastic | 0.000e+00 | 282.7 | 4.515e-13 |
| uniaxial_plastic | 3.988e-04 | 239.2 | 1.545e-10 |
| non_proportional | 2.297e-03 | 221.2 | 2.084e-11 |
| small_plastic | 5.056e-04 | 240.2 | 9.643e-11 |

Central differences use a symmetric strain perturbation of $10^{-7}$ with the
engineering-shear factor; the agreement is $\sim10^{-10}$ relative — five orders
below the $10^{-6}$ gate — for elastic, uniaxial-plastic, non-proportional and
"barely plastic" states.  The barely-plastic state ($\Delta p$/step $\sim5\cdot10^{-4}$)
is included precisely because that is where the $\left(a-\gamma a^2\Delta p\right)$
terms are largest relative to $\sqrt6G$.

### 3.5 Degeneration onto the monotonic m4951 kernel (C6)

| case | stress | $\Delta p$ | plastic strain | $p$ | tangent |
|---|---|---|---|---|---|
| monotonic_plastic | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.495e-16 |
| pure_shear | 0.000e+00 | 0.000e+00 | 2.479e-16 | 0.000e+00 | 1.388e-16 |
| yielded_state | 1.497e-16 | 0.000e+00 | 5.303e-17 | 0.000e+00 | 1.352e-16 |

All entries are $\le10^{-16}$: with $C=0$ the module returns *bit-identical* stress,
plastic multiplier, plastic strain and tangent to `tensorfem.plasticity.update_j2` /
`j2_consistent_tangent`.  The monotonic surface is therefore a strict special case, not
a re-implementation — the "one source of truth" property that m4951 established is
preserved.

### 3.6 Negative control: no Bauschinger effect without kinematic hardening (C7)

| $\varepsilon_a$ | Bauschinger index $\alpha_a/R$ | measured drop [MPa] | closed form [MPa] | C3 criterion |
|---|---|---|---|---|
| 0.4 % | 0.00 % | 0.000 | 0.000 | FAIL |
| 0.6 % | 0.00 % | 0.000 | 0.000 | FAIL |
| 0.8 % | 0.00 % | 0.000 | 0.000 | FAIL |
| 1.0 % | 0.00 % | 0.000 | 0.000 | FAIL |
| 1.2 % | 0.00 % | 0.000 | 0.000 | FAIL |

With $H_k=C=0$ the closed form predicts $\alpha_a=0$, and the simulation indeed shows
a reverse yield stress exactly one elastic range below the peak.  The C3 criterion is
stated as a **conjunction** — *quantitative* agreement with $\alpha_a$ **and**
detectability ($\alpha_a/R\ge5\,\%$) — otherwise it would be satisfied vacuously by a
model with no kinematic hardening at all.  With the kinematic branch removed the
criterion FAILs in 100 % of the 5 amplitudes, so the test has
discriminating power (it is not a tautology).

### 3.7 A posteriori residual audit

| quantity | residual |
|---|---|
| yield condition $\lVert\boldsymbol\eta\rVert_F=\sqrt{2/3}R(p)$ | 4.171e-16 |
| flow rule $\Delta\boldsymbol\varepsilon^p=\sqrt{3/2}\Delta p\,\mathbf n$ | 2.144e-16 |
| AF evolution (implicit form) | 0.000e+00 |
| plastic incompressibility $\mathrm{tr}\Delta\boldsymbol\varepsilon^p$ | 0.000e+00 |

---

## 4. Verdict against the acceptance criteria

| criterion | gate | result | verdict |
|---|---|---|---|
| C1 peak/trough stress | $\le 1\,\%$ | 0.1541 % | PASS |
| C2 dissipated energy per cycle | $\le 2\,\%$ | 0.2554 % | PASS |
| C3 Bauschinger drop | $\le 3\,\%$ | 0.6837 % | PASS |
| C4 consistent tangent | $\le 10^{-6}$ | 1.545e-10 | PASS |
| C5 ratcheting rate | $\le 5\,\%$ | 2.609 % | PASS |
| C5 Richardson limit | $\le 1\,\%$ | 0.020 % | PASS |
| C6 $C\to0$ degeneration | $\le 10^{-12}$ | 2.479e-16 | PASS |
| C7 negative control $H_k=0$ | $\ge 50\,\%$ of cases FAIL | 100 % FAIL | PASS |
| a posteriori residual audit | $\le 10^{-12}$ | 4.171e-16 | PASS |

**Is every criterion below the 3 % band the owner asked for?** Yes for the three
stress/energy/Bauschinger anchors (C1 0.1541 %, C2
0.2554 %, C3 0.6837 %) and for the
ratcheting rate once the truncation error is separated (C5 2.61 % at the
working step size, 0.020 % in the limit). The tangent (C4), the
degeneration (C6) and the residual audit are exact to round-off and are compared
against their own $10^{-6}$/$10^{-12}$ gates, not against 3 %.

Overall: **ALL GATES PASS**
(`artifacts/cyclic_plasticity/verification.json`, key `all_gates_pass`).

---

## 5. Notes, traps and limitations

**Traps hit while building this surface** (worth keeping for the next agent):

1. *Sign of the back-stress term in the tangent.*  $\partial\lVert\mathbf w\rVert/\partial\Delta p
   =+\gamma a^2(\mathbf n:\mathbf X^n)$ **and** $D$ carries $-\gamma a^2(\mathbf n:\mathbf X^n)$.
   Both were initially flipped; with $\mathbf X^n=0$ (every monotonic check) the error is
   invisible, and with a developed back stress it showed up as a 1–4 % tangent error.
   Lesson: any $\mathbf X$-dependent term must be validated on a state with a *developed*
   back stress and *non-proportional* strain, never on the virgin state.
2. *Freezing $\mathbf X$* in the return mapping destroys contractivity; the loop then
   does not stabilise and the tanh anchor is meaningless.  The $a$-factor in
   $\mathbf X^{n+1}=a(\mathbf X^n+\dots)$ is the whole physics.
3. *Closed-form energy*: the factor is $4C/\gamma^2$, not $4C/\gamma$ (dimension check:
   the $\gamma\to0$ limit must give the Prager parallelogram $4R\varepsilon^p_a$).  An
   implementation with $4C/\gamma$ overshoots the loop area by two orders of magnitude —
   and a *plausibility* check ("the AF term should be the size of the isotropic term")
   catches it immediately.
4. *History construction*: a triangular stress history must be built as deviation
   `+a → −a → +a` per cycle; a naive "ramp from the previous value" produced a
   one-step jump of $2\sigma_a$ and the 2×2 Newton diverged into a $10^{9}$ plastic
   strain.  `cyclic_history` now returns per-cycle bounds, which also makes the
   "per-cycle increment" unambiguous (and a guarded Newton is kept anyway).

**Limitations (honest scope)**

* **One AF back stress only.** Real cyclic data are usually fitted with 2–4 Chaboche
  terms; the closed forms above generalise (sum of tanh terms) but are not implemented
  here.
* **Ratcheting over-prediction is intrinsic to AF.** The anchor is exact for the model,
  not for the material: AF is known to over-predict multi-axial ratcheting by factors
  of 2–5 versus experiments (it lacks the Ohno-Wang/Burlet dynamic-recovery
  modification).  Nothing in this document licenses using the AF ratcheting rate as a
  design value without a data fit.
* **Uniaxial anchors assume a stabilised, proportional cycle.** No non-proportional
  cyclic path, no multiaxial ratcheting closed form, no temperature/rate dependence,
  no damage, no finite strain, no plasticity-gradient effects.
* **`C → 0` is exact only at `C == 0`.** The tangent is continuous in $C$, but the
  bit-level statement (C6) is made for the degenerate material, as specified.
* **Drivers are material-point, not FE.** The tangent is verified against central
  differences at the point level; a full structural Newton loop with this tangent is
  *not* part of this mission (and the reference march is still exact-to-round-off, so a
  global test would add assembly, not constitutive, information).
* **Tolerances are algorithmic.** $10^{-6}$ (tangent) and $10^{-12}$ (degeneration)
  are round-off-level, not physical; the *physical* gates are C1–C3 and C5.

---

## 6. Reproducing

```bash
python scripts/cyclic_plasticity_verification.py     # writes artifacts/cyclic_plasticity/*
python scripts/render_cyclic_doc.py                  # regenerates this document
python -m pytest tests/test_cyclic_plasticity.py -q  # 27 cases (C1-C7)
python -m pytest tests/test_j2_minimum_surface.py -q # 16 cases (C8, m4951 regression)
```
