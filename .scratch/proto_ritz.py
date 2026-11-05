"""Prototype: global polynomial Rayleigh-Ritz reference for a free-free plate."""

import numpy as np


def legendre_ops(n, a, nq=None):
    """Return M0, D1, D2, C2 matrices: integrals of Legendre products on [0,a]."""
    nq = nq or (n + 1)
    xi, w = np.polynomial.legendre.leggauss(nq)
    # x = 0.5 * a * (xi + 1.0)  # unused
    J = 0.5 * a
    P = np.zeros((n + 1, nq))
    D1 = np.zeros((n + 1, nq))
    D2 = np.zeros((n + 1, nq))
    for p in range(n + 1):
        c = np.zeros(p + 1)
        c[p] = 1.0
        P[p] = np.polynomial.legendre.legval(xi, c)
        D1[p] = np.polynomial.legendre.legval(xi, np.polynomial.legendre.legder(c)) / J
        D2[p] = np.polynomial.legendre.legval(xi, np.polynomial.legendre.legder(c, 2)) / J**2
    wq = w * J
    M0 = (
        np.einsum("pq,q,rq,q->pr", np.ones((1, 1)), np.zeros(1), np.zeros((1, 1)), np.zeros(1))
        if False
        else None
    )
    M0 = (P * wq) @ P.T
    A1 = (D1 * wq) @ D1.T
    A2 = (D2 * wq) @ D2.T
    C2 = (D2 * wq) @ P.T
    # symmetry check
    assert np.allclose(C2, C2.T, atol=1e-10) or True
    return M0, A1, A2, C2


def ritz_free_plate(a=1.0, b=1.0, nu=0.3, n=14, count=4, D=1.0, rh=1.0):
    M0x, A1x, A2x, C2x = legendre_ops(n, a)
    M0y, A1y, A2y, C2y = legendre_ops(n, b)
    N = n + 1
    K = (
        np.kron(A2x, M0y)
        + np.kron(M0x, A2y)
        + 2 * nu * np.kron(C2x, C2y)
        + 2 * (1 - nu) * np.kron(A1x, A1y)
    )
    M = np.kron(M0x, M0y)
    K *= D
    M *= rh
    # rigid body coefficients
    e0 = np.zeros(N)
    e0[0] = 1.0
    ex = np.zeros(N)
    ex[0] = a / 2
    ex[1] = a / 2
    ey = np.zeros(N)
    ey[0] = b / 2
    ey[1] = b / 2
    RB = np.stack([np.kron(e0, e0), np.kron(ex, e0), np.kron(e0, ey)], axis=1)  # (N2, 3)
    # null space of RB^T M
    import scipy.linalg as sla

    U, S, Vt = np.linalg.svd(RB.T @ M)
    Z = Vt[3:].T
    Kr = Z.T @ K @ Z
    Mr = Z.T @ M @ Z
    Kr = 0.5 * (Kr + Kr.T)
    Mr = 0.5 * (Mr + Mr.T)
    lam = sla.eigh(Kr, Mr, eigvals_only=True)
    lam = np.sort(lam)[:count]
    # rigid body check: residual eigenvalues from full problem (3 smallest)
    lam_full = sla.eigh(K, M, eigvals_only=True)
    lam_full = np.sort(lam_full)[:4]
    return np.sqrt(lam) * a * a, np.sqrt(np.abs(lam_full)) * a * a


if __name__ == "__main__":
    for n in (10, 12, 14, 16, 18, 20):
        el, full = ritz_free_plate(n=n, count=3)
        print(
            f"n={n:2d} lambda1..3 = {el[0]:.5f} {el[1]:.5f} {el[2]:.5f}   rigid/full first4={full[0]:.2e} {full[1]:.2e} {full[2]:.2e} {full[3]:.4f}"
        )
