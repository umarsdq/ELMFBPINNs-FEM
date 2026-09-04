"""Elliptic problems used by the matched ELM-FBPINN/FEM workflow.

The product-of-tanh hard constraint and its default scale of ``0.2`` follow
the two-dimensional problem definition in the ``elm-paper`` branch of
FBPINNs. The NumPy value-and-gradient form is the corresponding extension used
by the FEM assembly.

Source: https://github.com/benmoseley/FBPINNs/blob/elm-paper/elm-paper/problems.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from domains import RectangleDomain

@dataclass(frozen=True)
class EllipticProblem2D:
    """Interface for a homogeneous-Dirichlet scalar elliptic problem."""

    domain: RectangleDomain = field(default_factory=RectangleDomain)
    boundary_sd: float = 0.2

    def exact_solution(self, points):
        raise NotImplementedError

    def forcing(self, points):
        raise NotImplementedError

    def hard_constraint_jax(self, point):
        """Evaluate the FBPINNs ``elm-paper`` tanh boundary factor."""
        lower = jnp.asarray(self.domain.lower)
        upper = jnp.asarray(self.domain.upper)
        left = jnp.tanh((point - lower) / self.boundary_sd)
        right = jnp.tanh((upper - point) / self.boundary_sd)
        return jnp.prod(left * right)

    def hard_constraint_numpy(self, point):
        """Return the same boundary factor and its gradient for FEM assembly."""
        point = np.asarray(point, dtype=float)
        lower, upper = np.asarray(self.domain.lower), np.asarray(self.domain.upper)
        left = np.tanh((point - lower) / self.boundary_sd)
        right = np.tanh((upper - point) / self.boundary_sd)
        factors = left * right
        derivatives = ((1.0-left**2)*right-left*(1.0-right**2)) / self.boundary_sd
        value = float(np.prod(factors))
        gradient = np.array([derivatives[0]*factors[1], factors[0]*derivatives[1]])
        return value, gradient

    def strong_operator(self, basis_fn, point):
        """Apply the strong Poisson operator ``-Delta`` to a basis vector."""
        hessian = jax.jacfwd(jax.jacfwd(basis_fn))(point)
        return -(hessian[:, 0, 0] + hessian[:, 1, 1])

    def fem_local_matrix(self, basis, gradient, point):
        """Return the weak Poisson bilinear-form density."""
        del basis, point
        return gradient @ gradient.T


@dataclass(frozen=True)
class Poisson2D(EllipticProblem2D):
    """Manufactured Poisson problem with a separable sine solution.

    ``-Delta u = f``.

    with ``u = sin(omega*pi*x) sin(omega*pi*y)`` on the unit square after
    mapping from the configured rectangular domain.
    """

    omega: float = 1.0

    def exact_solution(self, points):
        points = jnp.asarray(points)
        lower = jnp.asarray(self.domain.lower)
        lengths = jnp.asarray(self.domain.lengths)
        local = (points - lower) / lengths
        return (jnp.sin(self.omega * jnp.pi * local[:, 0])
                * jnp.sin(self.omega * jnp.pi * local[:, 1]))

    def forcing(self, points):
        lengths = np.asarray(self.domain.lengths)
        scale = (self.omega * np.pi) ** 2 * np.sum(1.0 / lengths**2)
        return scale * self.exact_solution(points)


@dataclass(frozen=True)
class Helmholtz2D(Poisson2D):
    """Manufactured homogeneous-Dirichlet Helmholtz problem.
    
    ``-Delta u - k**2 u = f``.

    with ``u = sin(omega*pi*x) sin(omega*pi*y)`` on the unit square after
    mapping from the configured rectangular domain.
    """

    wavenumber: float = 2.0

    def forcing(self, points):
        lengths = np.asarray(self.domain.lengths)
        laplacian_scale = (self.omega * np.pi) ** 2 * np.sum(1.0 / lengths**2)
        return (laplacian_scale - self.wavenumber**2) * self.exact_solution(points)

    def strong_operator(self, basis_fn, point):
        poisson_part = super().strong_operator(basis_fn, point)
        return poisson_part - self.wavenumber**2 * basis_fn(point)

    def fem_local_matrix(self, basis, gradient, point):
        del point
        return gradient @ gradient.T - self.wavenumber**2 * np.outer(basis, basis)
