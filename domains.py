"""Domain and matched-grid constructions for the ELM-FBPINN/FEM comparison.

The ELM decomposition directly uses ``get_subdomain_ws`` and
``RectangularDecompositionND`` from the ``elm-paper`` branch of FBPINNs:
https://github.com/benmoseley/FBPINNs/tree/elm-paper
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fbpinns.constants import get_subdomain_ws
from fbpinns.decompositions import RectangularDecompositionND

@dataclass(frozen=True)
class RectangleDomain:
    """Axis-aligned two-dimensional physical domain."""

    lower: tuple[float, float] = (0.0, 0.0)
    upper: tuple[float, float] = (1.0, 1.0)

    @property
    def lengths(self) -> np.ndarray:
        return np.subtract(self.upper, self.lower)

    def tensor_grid(self, n_per_dim: int):
        """Return an endpoint-inclusive tensor grid for evaluation."""
        axes = [
            np.linspace(lower, upper, n_per_dim)
            for lower, upper in zip(self.lower, self.upper)
        ]
        X, Y = np.meshgrid(*axes, indexing="ij")
        return X, Y, np.column_stack([X.ravel(), Y.ravel()])


@dataclass(frozen=True)
class MatchedGeometry:
    """Geometry shared by one matched ELM-FBPINN/FEM experiment.

    ``J`` is the number of ELM subdomains per direction and ``order`` is the
    FEM order parameter ``p``. Both methods use ``(J*p)**2`` matched points
    and solved coefficients.
    """

    domain: RectangleDomain = RectangleDomain()
    J: int = 4
    order: int = 1
    overlap: float = 1.999

    @property
    def element_width(self) -> np.ndarray:
        return self.domain.lengths / self.J

    @property
    def shifted_lower(self) -> np.ndarray:
        return np.asarray(self.domain.lower) - 0.5 * self.element_width

    @property
    def shifted_upper(self) -> np.ndarray:
        return np.asarray(self.domain.upper) + 0.5 * self.element_width

    def elm_centres(self) -> list[np.ndarray]:
        return [np.linspace(a, b, self.J) for a, b in zip(self.domain.lower, self.domain.upper)]

    def elm_decomposition(self):
        """Build the endpoint-centred decomposition with FBPINNs utilities."""
        centres = self.elm_centres()
        widths = get_subdomain_ws(centres, self.overlap)
        static, trainable = RectangularDecompositionND.init_params(centres,
                                                                  widths,
                                                                  unnorm=(0.0, 1.0))
        return {"static": {"decomposition": static}, "trainable": {"decomposition": trainable}}

    def shifted_fem_mesh(self):
        """Return the half-element-extended tensor-product FEM mesh."""
        n_nodes = (self.J + 1) * self.order + 1
        xs = np.linspace(self.shifted_lower[0], self.shifted_upper[0], n_nodes)
        ys = np.linspace(self.shifted_lower[1], self.shifted_upper[1], n_nodes)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        elements = []
        for ex in range(self.J + 1):
            for ey in range(self.J + 1):
                elements.append([(ex*self.order+i)*n_nodes+ey*self.order+j
                                 for i in range(self.order+1) for j in range(self.order+1)])
        return nodes, np.asarray(elements, dtype=int)

    def active_fem_indices(self, nodes: np.ndarray):
        """Return IDs of shifted-mesh FEM nodes inside the physical domain."""
        lower, upper = np.asarray(self.domain.lower), np.asarray(self.domain.upper)
        tolerance = 1e-12 * np.max(self.domain.lengths)
        return np.flatnonzero(np.all((nodes > lower+tolerance)
                                     & (nodes < upper-tolerance), axis=1))

    def matched_points(self) -> np.ndarray:
        """Use the active shifted-FEM nodes as ELM collocation points."""
        nodes, _ = self.shifted_fem_mesh()
        return nodes[self.active_fem_indices(nodes)]
