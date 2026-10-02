# Arc-length limit-point qualification and shell integration boundary

`arc_length` implements a spherical Crisfield continuation constraint with an
augmented Newton corrector, orientation-continuous predictor, adaptive step
size, failed-step rollback and schema-tagged JSON restart. Its residual
interface accepts the internal force and consistent tangent already produced
by `general_shell_nonlinear`; no shell-specific secant is introduced.
`general_shell_arc_problem` performs the constrained-DOF reduction and exposes
the assembled consistent shell tangent directly to the continuation engine.

Algorithm qualification uses the classical two-bar von Mises shallow arch:
half-span `a=1`, rise `h=0.2`, and bar axial rigidity `EA=1000`. For downward
apex displacement `w`, current height `y=h-w`, initial length
`l0=sqrt(a^2+h^2)`, and current length `l=sqrt(a^2+y^2)`, the exact branch is

`P(w)=2 EA/l0 * (l0/l - 1) y`.

The positive limit point is independent of a numerical reference:
`l^3=l0 a^2`, which explicitly determines `w_limit` and `P_limit`. Tests must
cross that point and contain a decreasing-load/increasing-displacement branch.
Arc sizes 0.01 and 0.005 must predict peak load within 3% of one another and of
the closed form. A split checkpoint/restart path must reproduce a continuous
35-step run.

| arc size | computed peak displacement | displacement error | peak load | load error |
|---:|---:|---:|---:|---:|
| 0.01 | 0.088805 | 4.126% | 2.956432 | 0.138% |
| 0.005 | 0.083783 | 1.761% | 2.959763 | 0.025% |

Only the 0.005 path qualifies the peak coordinates; the coarser displacement
miss is retained as convergence evidence. Both paths continue past the maximum
onto decreasing and eventually negative load rather than stopping at the
limit point.

The von Mises truss/arch is a recognized snap-through algorithm benchmark, not
a claim that a published snap-through shell has passed. It qualifies the Riks
continuation engine that can consume the general shell residual/tangent.
Published shell snap-through geometry, follower loads and branch switching
remain separate capability gates.

Algorithm reference: M. A. Crisfield, “A fast incremental/iterative solution
procedure that handles snap-through,” *Computers & Structures* 13 (1981),
55--62, DOI `10.1016/0045-7949(81)90108-5`.
