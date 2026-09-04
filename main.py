"""Command-line interface for the matched ELM-FBPINN/FEM comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from methods import (
    DEFAULT_ITERATION_LIMIT,
    DEFAULT_TEST_GRID,
    ComparisonConfig,
    run_comparison,
)
from plots import (
    plot_conditioning,
    plot_joint_scaling,
    plot_matrix_patterns,
    plot_overlap,
    plot_polynomial_sampling,
    plot_refinement,
    plot_solution_fields,
)
from problems import Helmholtz2D, Poisson2D


FIGURE_DIR = Path("./figures")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",
                        choices=("comparison", "thesis"),
                        default="comparison",
                        help="Run one configurable comparison or the thesis replication suite.")
    parser.add_argument("--J", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--overlap", type=float, default=1.999)
    parser.add_argument("--feature-family",
                        choices=("random_tanh", "polynomial"),
                        default="random_tanh")
    parser.add_argument("--sampling-factor",
                        type=int,
                        default=1,
                        help="Collocation points per direction divided by J*p.")
    parser.add_argument("--iteration-limit", type=int, default=DEFAULT_ITERATION_LIMIT)
    parser.add_argument("--test-grid", type=int, default=DEFAULT_TEST_GRID)
    parser.add_argument("--problem", choices=("poisson", "helmholtz"), default="poisson")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--wavenumber", type=float, default=2.0)
    parser.add_argument("--boundary-sd", type=float, default=0.2)
    parser.add_argument("--condition",
                        action="store_true",
                        help="Compute dense matrix condition numbers (disabled by default).")
    return parser.parse_args()


def build_problem(args):
    values = {"omega": args.omega, "boundary_sd": args.boundary_sd}
    if args.problem == "helmholtz":
        return Helmholtz2D(wavenumber=args.wavenumber, **values)
    return Poisson2D(**values)


def comparison_config(args):
    return ComparisonConfig(J_values=tuple(args.J),
                            orders=tuple(args.orders),
                            seeds=tuple(args.seeds),
                            test_grid=args.test_grid,
                            overlap=args.overlap,
                            feature_family=args.feature_family,
                            sampling_factor=args.sampling_factor,
                            iteration_limit=args.iteration_limit,
                            compute_condition=args.condition)


def run_case(name, problem, config, include_fem=True):
    print(f"\nRunning {name}")
    results = run_comparison(problem, config, include_fem)
    summary = results.summary()
    summary.insert(0, "case", name)
    return results, summary


def run_thesis_suite():
    """Regenerate the principal thesis data and figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = dict(test_grid=51, iteration_limit=2000, compute_condition=False)
    frames = []

    refinement = ComparisonConfig(J_values=(2, 4, 6, 8),
                                  orders=(1, 3),
                                  seeds=(0, 1, 2),
                                  **base)
    poisson_results, poisson_summary = run_case("poisson_refinement",
                                                Poisson2D(),
                                                refinement)
    frames.append(poisson_summary)
    plot_refinement(poisson_summary, FIGURE_DIR, "poisson_error_refinement")
    plot_solution_fields(poisson_results, FIGURE_DIR)
    plot_matrix_patterns(poisson_results, FIGURE_DIR)

    _, helmholtz_summary = run_case("helmholtz_refinement",
                                    Helmholtz2D(wavenumber=2.0),
                                    refinement)
    frames.append(helmholtz_summary)
    plot_refinement(helmholtz_summary, FIGURE_DIR, "helmholtz_error_refinement")

    conditioning = ComparisonConfig(J_values=(2, 4, 6, 8),
                                    orders=(1, 3),
                                    seeds=(0,),
                                    test_grid=51,
                                    compute_condition=True)
    _, conditioning_summary = run_case("poisson_conditioning",
                                       Poisson2D(),
                                       conditioning)
    frames.append(conditioning_summary)
    plot_conditioning(conditioning_summary, FIGURE_DIR)

    polynomial_frames = []
    for sampling_factor in (1, 2):
        config = ComparisonConfig(J_values=(2, 4, 6, 8),
                                  orders=(3,),
                                  seeds=(0,),
                                  feature_family="polynomial",
                                  sampling_factor=sampling_factor,
                                  **base)
        _, summary = run_case(f"polynomial_sampling_{sampling_factor}",
                              Poisson2D(),
                              config,
                              include_fem=False)
        frames.append(summary)
        polynomial_frames.append(summary)
    plot_polynomial_sampling(pd.concat(polynomial_frames, ignore_index=True), FIGURE_DIR)

    overlaps = (1.01, 1.05, 1.25, 1.5, 1.75, 1.999, 2.0, 2.0001, 2.25, 2.5, 3.0)
    overlap_frames = []
    for overlap in overlaps:
        config = ComparisonConfig(J_values=(4,),
                                  orders=(1, 3),
                                  seeds=(0,),
                                  overlap=overlap,
                                  **base)
        _, summary = run_case(f"overlap_{overlap:g}",
                              Poisson2D(),
                              config,
                              include_fem=False)
        frames.append(summary)
        overlap_frames.append(summary)
    plot_overlap(pd.concat(overlap_frames, ignore_index=True), FIGURE_DIR)

    scaling_frames = []
    for omega, J in ((1, 2), (2, 4), (4, 8)):
        config = ComparisonConfig(J_values=(J,),
                                  orders=(3,),
                                  seeds=(0, 1, 2),
                                  **base)
        _, summary = run_case(f"joint_scaling_{omega:g}",
                              Poisson2D(omega=omega),
                              config)
        frames.append(summary)
        scaling_frames.append(summary)
    plot_joint_scaling(pd.concat(scaling_frames, ignore_index=True), FIGURE_DIR)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nThesis figures saved to {FIGURE_DIR}")
    return combined


def main():
    args = parse_args()
    if args.study == "thesis":
        return run_thesis_suite()

    results = run_comparison(build_problem(args), comparison_config(args))
    print(results.summary().to_string(index=False))
    return results


if __name__ == "__main__":
    main()
