"""ELM-FBPINN and shifted FEM discretisations plus comparison workflows.

The ELM-FBPINN implementation reuses decomposition, window and ELM feature
components from the ``elm-paper`` branch of FBPINNs. Its residual assembly is
a compact, problem-specific adaptation of that branch's linear ELM-FBPINN
workflow. The FEM element integration and local-to-global assembly were
developed from Sean De Marco's Julia ``2Dpois_FEM.jl`` baseline and extended
here to the shifted, hard-constrained ``Q_p`` construction. Full links and the
scope of each adaptation are recorded in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from domains import MatchedGeometry
from problems import EllipticProblem2D, Poisson2D
from fbpinns import windows
from fbpinns.decompositions import RectangularDecompositionND
from elm.elms import ELM

jax.config.update("jax_enable_x64", True)

DEFAULT_TEST_GRID = 51
DEFAULT_ITERATION_LIMIT = 2000
FEATURE_FAMILIES = {"random_tanh", "polynomial"}

def validate_config(J_values, orders, feature_family=None):
    """Validate settings shared by the matched solver configurations."""
    if any(J < 2 for J in J_values):
        raise ValueError("Every J value must be at least 2.")
    if any(order < 1 or order % 2 == 0 for order in orders):
        raise ValueError("The shifted match requires odd orders (1, 3, 5, ...).")
    if feature_family is not None and feature_family not in FEATURE_FAMILIES:
        raise ValueError(f"feature_family must be one of {FEATURE_FAMILIES}.")

def make_test_grid(problem: EllipticProblem2D, n_per_dim=DEFAULT_TEST_GRID):
    X, Y, points = problem.domain.tensor_grid(n_per_dim)
    exact = np.asarray(problem.exact_solution(points)).reshape(-1)
    return X, Y, points, exact

def error_metrics(prediction, exact):
    prediction, exact = np.asarray(prediction), np.asarray(exact)
    standard_deviation = max(float(np.std(exact)), np.finfo(float).eps)
    return {"normalised_l1": float(np.mean(np.abs(prediction-exact))/standard_deviation),
            "relative_l2": float(np.linalg.norm(prediction-exact)/np.linalg.norm(exact)),
            "maximum": float(np.max(np.abs(prediction - exact)))}

def threshold_to_csr(matrix, tolerance):
    """Remove floating-point remnants of support zeros and convert to CSR."""
    values = np.array(matrix, copy=True)
    values[np.abs(values) < tolerance] = 0.0
    return sps.csr_matrix(values)

def relative_residual(matrix, coefficients, rhs):
    denominator = max(np.linalg.norm(rhs), np.finfo(float).eps)
    return float(np.linalg.norm(matrix @ coefficients - rhs) / denominator)

def _solve_lsqr(matrix, rhs, iteration_limit, compute_condition):
    start = time.perf_counter()
    result = spla.lsqr(matrix,
                       rhs,
                       atol=1e-12,
                       btol=1e-12,
                       iter_lim=iteration_limit)
    solve_time = time.perf_counter() - start
    coefficients = result[0]
    info = {
        "solver": "LSQR",
        "iterations": int(result[2]),
        "status": int(result[1]),
        "relative_residual": relative_residual(matrix, coefficients, rhs),
        "condition": (float(np.linalg.cond(matrix.toarray()))
                      if compute_condition else np.nan),
        "solve_time": solve_time,
    }
    return coefficients, info

class Polynomial:
    """Tensor-product polynomial feature map.

    For a two-dimensional ``order=p`` experiment this produces the ``p**2``
    monomials ``x**i*y**j`` with ``0 <= i,j < p``.
    """

    @staticmethod
    def init_params(input_dimension, order):
        """Store tensor-product exponents in the original total-degree order."""
        exponents = sorted(product(range(order), repeat=input_dimension),
                           key=lambda powers: (sum(powers), tuple(-p for p in powers)))
        return {"exponents": jnp.asarray(exponents[1:]).reshape(-1, input_dimension)}

    @staticmethod
    def n_terms(input_dimension, order):
        return order**input_dimension

    @staticmethod
    def basis_fn(params, point):
        """Evaluate the constant term followed by all tensor-product monomials."""
        monomials = jnp.prod(point**params["exponents"], axis=1)
        return jnp.concatenate((jnp.ones(1, dtype=point.dtype), monomials))



@dataclass(frozen=True)
class ELMConfig:
    """Configuration for the native strong-form ELM-FBPINN."""

    J: int = 4
    order: int = 1
    overlap: float = 1.999
    feature_family: str = "random_tanh"
    seed: int = 0
    weight_scale: float = 1.0
    iteration_limit: int = DEFAULT_ITERATION_LIMIT
    zero_tolerance: float = 1e-11
    compute_condition: bool = False
    collocation_points_per_dim: int | None = None

    def __post_init__(self):
        validate_config((self.J,), (self.order,), self.feature_family)
        if (self.collocation_points_per_dim is not None
                and self.collocation_points_per_dim < 1):
            raise ValueError("collocation_points_per_dim must be positive.")

class ELMFBPINN:
    """Strong-form ELM-FBPINN built with FBPINNs/ELM components.

    ``ELM.init_params`` and ``ELM.basis_fn`` provide the frozen tanh features.
    FBPINNs supplies the rectangular decomposition, local normalisation and
    cosine windows. The partition-of-unity and residual assembly below adapt
    the workflow in ``elm/trainers.py`` to expose the matrix used in this
    controlled comparison.

    Source: https://github.com/benmoseley/FBPINNs/tree/elm-paper
    """

    def __init__(self, problem=None, config=None):
        self.problem = problem or Poisson2D()
        self.config = config or ELMConfig()
        self.geometry = MatchedGeometry(self.problem.domain,
                                        self.config.J,
                                        self.config.order,
                                        self.config.overlap)
        self.decomposition = self.geometry.elm_decomposition()
        self.parameters = self.initialise_features()
        self.system = None
        self.coefficients = None
        self.solve_info = None

    @property
    def K(self):
        if self.config.feature_family == "polynomial":
            return Polynomial.n_terms(2, self.config.order)
        return self.config.order**2

    @property
    def n_coefficients(self):
        return self.config.J**2 * self.K

    def initialise_features(self):
        if self.config.feature_family == "polynomial":
            return Polynomial.init_params(2, self.config.order)

        if self.K == 1:
            return None  # No features, just a constant basis function

        keys = jax.random.split(jax.random.PRNGKey(self.config.seed), self.config.J**2)
        static, _ = jax.vmap(ELM.init_params, in_axes=(0, None, None))(keys,
                                                                      [2, self.K-1, 1],
                                                                      self.config.weight_scale)
        return static

    def partition_basis(self, point):
        """Assemble the local FBPINNs windows and ELM features at one point."""
        params = self.decomposition["static"]["decomposition"]["subdomain"]["params"]
        xmins, xmaxs = params[:2]
        window_values = jax.vmap(lambda xmin, xmax:
                                 windows.cosine(xmin, xmax, point).squeeze())(xmins, xmaxs)
        local_points = RectangularDecompositionND.norm_fn(self.decomposition, point)

        if self.config.feature_family == "polynomial":
            features = jax.vmap(lambda local_point:
                                Polynomial.basis_fn(self.parameters, local_point))(local_points)
        elif self.K == 1:
            features = jnp.ones((self.config.J**2, 1))
        else:
            def elm_basis(params, local_point):
                params = {"static": {"network": {"subdomain": params}}}
                return ELM.basis_fn(params, local_point)

            features = jax.vmap(elm_basis)(self.parameters, local_points)

        normalised_windows = window_values / jnp.sum(window_values)
        return (normalised_windows[:, None] * features).reshape(-1)

    def basis(self, point):
        return self.problem.hard_constraint_jax(point)*self.partition_basis(point)

    def basis_matrix(self, points):
        return np.asarray(jax.vmap(self.basis)(jnp.asarray(points)))

    def physics_row(self, point):
        return self.problem.strong_operator(self.basis, point)

    def collocation_points(self):
        n = self.config.collocation_points_per_dim
        if n is None:
            return self.geometry.matched_points()
        lower, upper = np.asarray(self.problem.domain.lower), np.asarray(self.problem.domain.upper)
        axes = [a + (np.arange(n) + 0.5) * (b-a) / n for a, b in zip(lower, upper)]
        X, Y = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([X.ravel(), Y.ravel()])

    def assemble(self):
        """Assemble strong-form rows using the FBPINNs basis and JAX AD.

        This is a compact adaptation of the linear assembly in
        ``elm/trainers.py`` rather than a call to the full trainer, allowing
        the matched collocation matrix to remain directly accessible.
        """
        assembly_fn = jax.jit(lambda x: (jax.vmap(self.physics_row)(x),
                                         self.problem.forcing(x)))
        points = self.collocation_points()
        points_jax = jnp.asarray(points)
        warmup_rows, warmup_rhs = assembly_fn(points_jax)
        warmup_rows.block_until_ready()
        warmup_rhs.block_until_ready() # Warmup to compile the JIT function (time only the 2nd call)

        start = time.perf_counter()
        rows, rhs = assembly_fn(points_jax)
        rows.block_until_ready()
        rhs.block_until_ready()
        matrix = threshold_to_csr(rows, self.config.zero_tolerance)
        rhs = np.asarray(rhs).reshape(-1)
        self.system = {
            "matrix": matrix,
            "rhs": rhs,
            "points": points,
            "assembly_time": time.perf_counter() - start,
            "metadata": {
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "nnz": matrix.nnz,
                "coefficients": self.n_coefficients,
                "J": self.config.J,
                "order": self.config.order,
                "K": self.K,
                "feature_family": self.config.feature_family,
                "collocation_points_per_dim": int(np.sqrt(matrix.shape[0])),
            },
        }
        return self.system

    def solve(self, iteration_limit=None):
        if self.system is None:
            self.assemble()
        self.coefficients, self.solve_info = _solve_lsqr(self.system["matrix"],
                                                         self.system["rhs"],
                                                         iteration_limit or self.config.iteration_limit,
                                                         self.config.compute_condition)
        return self.coefficients, self.solve_info

    def evaluate(self, points, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients
        if coefficients is None:
            raise ValueError("Solve the system or supply coefficients before evaluation.")
        return self.basis_matrix(points) @ np.asarray(coefficients)

    def run(self, test_points, exact):
        self.assemble()
        self.solve()
        prediction = self.evaluate(test_points)
        return RunResult.from_solver("ELM-FBPINN",
                                     self.config.J,
                                     self.config.order,
                                     self.config.seed,
                                     self,
                                     prediction,
                                     exact)

@dataclass(frozen=True)
class FEMConfig:
    """Configuration for the half-element-shifted ``Q_p`` FEM."""

    J: int = 4
    order: int = 1
    iteration_limit: int = DEFAULT_ITERATION_LIMIT
    compute_condition: bool = False

    def __post_init__(self):
        validate_config((self.J,), (self.order,))

class FEM:
    """Hard-constrained ``Q_p`` FEM on a half-element-extended mesh.

    The reference-element quadrature and local-to-global assembly follow the
    structure of Sean De Marco's Julia ``2Dpois_FEM.jl`` Poisson baseline.
    This implementation rewrites and extends that ``Q_1`` code to arbitrary
    odd ``Q_p`` order, a shifted mesh, sparse assembly, the common hard
    boundary factor and LSQR.

    Source: https://github.com/SeanDeMarco/Physics-Informed-Extreme-Learning-Machines/blob/main/2Dpois_FEM.jl
    """

    def __init__(self, problem=None, config=None):
        self.problem = problem or Poisson2D()
        self.config = config or FEMConfig()
        self.geometry = MatchedGeometry(self.problem.domain,
                                        self.config.J,
                                        self.config.order,
                                        overlap=1.999)
        self.system = None
        self.coefficients = None
        self.solve_info = None

    @staticmethod
    def lagrange_1d(xi, order):
        nodes = np.linspace(-1.0, 1.0, order + 1)
        values = np.ones(order + 1)
        derivatives = np.zeros(order + 1)
        for i in range(order + 1):
            for j in range(order + 1):
                if j != i:
                    values[i] *= (xi-nodes[j])/(nodes[i]-nodes[j])
            for k in range(order + 1):
                if k == i:
                    continue
                term = 1.0 / (nodes[i] - nodes[k])
                for j in range(order + 1):
                    if j != i and j != k:
                        term *= (xi-nodes[j])/(nodes[i]-nodes[j])
                derivatives[i] += term
        return values, derivatives

    @classmethod
    def shape_functions(cls, xi, eta, order):
        lx, dlx = cls.lagrange_1d(xi, order)
        ly, dly = cls.lagrange_1d(eta, order)
        basis = np.outer(lx, ly).ravel()
        reference_gradient = np.column_stack([np.outer(dlx, ly).ravel(),
                                              np.outer(lx, dly).ravel()])
        return basis, reference_gradient

    def create_mesh(self):
        return self.geometry.shifted_fem_mesh()

    def assemble(self):
        """Integrate element matrices and assemble the global sparse system.

        Adapted from the element-quadrature and assembly pattern in Sean De
        Marco's ``2Dpois_FEM.jl`` and generalised as described in the class
        docstring.
        """
        start = time.perf_counter()
        J, order = self.config.J, self.config.order
        nodes, elements = self.create_mesh()
        active = self.geometry.active_fem_indices(nodes)
        active_nodes = nodes[active]
        dof_map = -np.ones(len(nodes), dtype=int)
        dof_map[active] = np.arange(len(active))

        lower = np.asarray(self.problem.domain.lower)
        upper = np.asarray(self.problem.domain.upper)
        hx, hy = self.geometry.element_width
        gauss_order = max(order + 2, 4)
        quadrature_points, quadrature_weights = np.polynomial.legendre.leggauss(gauss_order)
        rows, columns, values = [], [], []
        rhs = np.zeros(len(active))

        for element in elements:
            x0, y0 = nodes[element[0]]
            x_left, x_right = max(x0, lower[0]), min(x0 + hx, upper[0])
            y_left, y_right = max(y0, lower[1]), min(y0 + hy, upper[1])
            if x_left >= x_right or y_left >= y_right:
                continue

            n_local = (order + 1)**2
            local_matrix = np.zeros((n_local, n_local))
            local_rhs = np.zeros(n_local)
            for xi, wx in zip(quadrature_points, quadrature_weights):
                for eta, wy in zip(quadrature_points, quadrature_weights):
                    x = 0.5*(x_left+x_right)+0.5*(x_right-x_left)*xi
                    y = 0.5*(y_left+y_right)+0.5*(y_right-y_left)*eta
                    reference_x, reference_y = 2*(x-x0)/hx-1, 2*(y-y0)/hy-1
                    basis, reference_gradient = self.shape_functions(reference_x,
                                                                     reference_y,
                                                                     order)
                    gradient = reference_gradient*np.array([2/hx, 2/hy])
                    point = np.array([x, y])
                    boundary, boundary_gradient = self.problem.hard_constraint_numpy(point)
                    constrained_basis = boundary * basis
                    constrained_gradient = (boundary*gradient
                                            + basis[:, None]*boundary_gradient[None, :])
                    weight = wx*wy*(x_right-x_left)*(y_right-y_left)/4
                    local_matrix += weight*self.problem.fem_local_matrix(constrained_basis,
                                                                         constrained_gradient,
                                                                         point)
                    forcing = float(np.asarray(self.problem.forcing(point.reshape(1, 2)))[0])
                    local_rhs += weight * forcing * constrained_basis

            dofs = dof_map[element]
            for local_i, global_i in enumerate(dofs):
                if global_i < 0:
                    continue
                rhs[global_i] += local_rhs[local_i]
                for local_j, global_j in enumerate(dofs):
                    if global_j >= 0:
                        rows.append(global_i)
                        columns.append(global_j)
                        values.append(local_matrix[local_i, local_j])

        matrix = sps.csr_matrix((values, (rows, columns)), shape=(len(active), len(active)))
        self.system = {
            "matrix": matrix,
            "rhs": rhs,
            "nodes": nodes,
            "elements": elements,
            "interior": active,
            "dof_map": dof_map,
            "assembly_time": time.perf_counter() - start,
            "metadata": {
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "nnz": matrix.nnz,
                "coefficients": len(active),
                "J": J,
                "order": order,
                "K": order**2,
                "active_nodes": active_nodes,
            },
        }
        return self.system

    def solve(self, iteration_limit=None):
        if self.system is None:
            self.assemble()
        self.coefficients, self.solve_info = _solve_lsqr(self.system["matrix"],
                                                         self.system["rhs"],
                                                         iteration_limit or self.config.iteration_limit,
                                                         self.config.compute_condition)
        return self.coefficients, self.solve_info

    def evaluate(self, points, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients
        if coefficients is None or self.system is None:
            raise ValueError("Assemble and solve the system before evaluation.")

        J, order = self.config.J, self.config.order
        local = (np.asarray(points)-self.geometry.shifted_lower)/self.geometry.element_width
        cells = np.clip(np.floor(local).astype(int), 0, J)
        reference = 2.0 * (local - cells) - 1.0
        mesh_width = (J + 1) * order + 1
        dof_map = self.system["dof_map"]
        prediction = np.zeros(len(points))

        for q, ((ex, ey), (xi, eta), point) in enumerate(zip(cells, reference, points)):
            basis, _ = self.shape_functions(xi, eta, order)
            nodes = np.array([
                (ex*order+i)*mesh_width+ey*order+j
                for i in range(order+1) for j in range(order+1)
            ])
            dofs = dof_map[nodes]
            active = dofs >= 0
            boundary, _ = self.problem.hard_constraint_numpy(point)
            prediction[q] = boundary*basis[active]@np.asarray(coefficients)[dofs[active]]
        return prediction

    def run(self, test_points, exact):
        self.assemble()
        self.solve()
        prediction = self.evaluate(test_points)
        return RunResult.from_solver("Shifted FEM",
                                     self.config.J,
                                     self.config.order,
                                     None,
                                     self,
                                     prediction,
                                     exact)


@dataclass
class RunResult:
    method: str
    J: int
    order: int
    seed: int | None
    solver: object
    prediction: np.ndarray
    metrics: dict
    assembly_time: float
    solve_time: float
    total_time: float
    solver_info: dict

    @classmethod
    def from_solver(cls, method, J, order, seed, solver, prediction, exact):
        assembly_time = solver.system["assembly_time"]
        solve_time = solver.solve_info["solve_time"]
        return cls(method, J, order, seed, solver, np.asarray(prediction),
                   error_metrics(prediction, exact), assembly_time, solve_time,
                   assembly_time+solve_time, dict(solver.solve_info))

    @property
    def matrix(self):
        return self.solver.system["matrix"]

@dataclass(frozen=True)
class ComparisonConfig:
    """Controls a matched sweep over spatial resolution and local order."""

    J_values: tuple[int, ...] = (2, 4, 6, 8)
    orders: tuple[int, ...] = (1, 3)
    seeds: tuple[int, ...] = (0, 1, 2)
    test_grid: int = DEFAULT_TEST_GRID
    overlap: float = 1.999
    feature_family: str = "random_tanh"
    sampling_factor: int = 1
    zero_tolerance: float = 1e-11
    iteration_limit: int = DEFAULT_ITERATION_LIMIT
    compute_condition: bool = False

    def __post_init__(self):
        validate_config(self.J_values, self.orders, self.feature_family)
        if self.sampling_factor < 1:
            raise ValueError("sampling_factor must be a positive integer.")

@dataclass
class ComparisonResults:
    problem: EllipticProblem2D
    config: ComparisonConfig
    X: np.ndarray
    Y: np.ndarray
    test_points: np.ndarray
    exact: np.ndarray
    elm_runs: list[RunResult]
    fem_runs: list[RunResult]

    def summary(self):
        rows = []
        for run in self.elm_runs + self.fem_runs:
            rows.append({
                "method": run.method,
                "problem": type(self.problem).__name__,
                "omega": getattr(self.problem, "omega", np.nan),
                "wavenumber": getattr(self.problem, "wavenumber", np.nan),
                "J": run.J,
                "seed": run.seed,
                "order": run.order,
                "K": run.order**2,
                "overlap": self.config.overlap,
                "sampling_factor": (
                    self.config.sampling_factor if run.method == "ELM-FBPINN" else 1
                ),
                "coefficients": run.solver.system["metadata"]["coefficients"],
                "rows": run.matrix.shape[0],
                "columns": run.matrix.shape[1],
                "nnz": run.matrix.nnz,
                **run.metrics,
                "assembly_time": run.assembly_time,
                "solve_time": run.solve_time,
                "total_time": run.total_time,
                "iterations": run.solver_info.get("iterations", np.nan),
                "relative_residual": run.solver_info.get("relative_residual", np.nan),
                "condition": run.solver_info.get("condition", np.nan),
                "feature_family": run.solver.system["metadata"].get("feature_family",
                                                                      "finite_element"),
            })
        return pd.DataFrame(rows)

    def elm_run(self, J, order=1, seed=0):
        return next(run for run in self.elm_runs
                    if run.J == J and run.order == order and run.seed == seed)

    def fem_run(self, J, order=1):
        return next(run for run in self.fem_runs if run.J == J and run.order == order)

def run_comparison(problem=None, config=None, include_fem=True):
    """Run a matched ELM-FBPINN/FEM sweep."""
    problem = problem or Poisson2D()
    config = config or ComparisonConfig()
    X, Y, test_points, exact = make_test_grid(problem, config.test_grid)
    elm_runs, fem_runs = [], []

    for order in config.orders:
        for J in config.J_values:
            if include_fem:
                fem = FEM(problem,
                          FEMConfig(J=J,
                                    order=order,
                                    iteration_limit=config.iteration_limit,
                                    compute_condition=config.compute_condition))
                fem_runs.append(fem.run(test_points, exact))

            for seed in config.seeds:
                elm = ELMFBPINN(problem,
                                ELMConfig(J=J,
                                          order=order,
                                          overlap=config.overlap,
                                          feature_family=config.feature_family,
                                          seed=seed,
                                          iteration_limit=config.iteration_limit,
                                          compute_condition=config.compute_condition,
                                          zero_tolerance=config.zero_tolerance,
                                          collocation_points_per_dim=(
                                              None if config.sampling_factor == 1
                                              else config.sampling_factor * J * order)))
                elm_runs.append(elm.run(test_points, exact))

    return ComparisonResults(problem, config, X, Y, test_points, exact, elm_runs, fem_runs)
