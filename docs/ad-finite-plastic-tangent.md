# AD finite-plastic algorithmic tangent

PyTorch differentiates the element residual through the accepted adaptive
substep tree and multiplicative returns. This replaces 49 forward material
paths per TET4 with one AD trace plus explicit response/evidence paths. The
Richardson tangent remains the oracle; non-coaxial error must remain below 3%,
Newton iterations may not increase, and wall-time/path reductions are measured.

Accept/reject decisions are discrete and held fixed during a Jacobian, so the
tangent is piecewise consistent. Spectral derivatives can be ill-conditioned
at repeated principal stretches; non-finite tangents fail closed. This is a
validated AD algorithmic tangent, not a closed-form implicit derivative.
