"""Steps of the computational experiment, built from the other modules.

The notebook calls these functions, so it holds no core logic. Each
function runs one step of the experiment and returns plain results or a
table.
"""

import numpy as np
import pandas as pd

from hyperfit import (
    data,
    evaluation,
    fem,
    fitting,
    kinematics,
    mechanics,
    models,
    parameters,
)

LINEAR = "Linear least squares"
LM = "Levenberg-Marquardt"
CMA_ES = "CMA-ES"


def fit_with_all_optimizers(
    variant: models.ModelVariant,
    residual: fitting.ResidualFunction,
    specs: list[parameters.ParameterSpec],
    jacobian: fitting.ResidualFunction | None = None,
) -> dict[str, fitting.FitResult]:
    """Fits one variant with every optimizer that applies.

    Linear least squares applies only to the Yeoh family, because only the
    Yeoh residual is linear in theta.

    Args:
        variant: The model variant.
        residual: The residual function.
        specs: The parameter specifications.
        jacobian: An optional Jacobian function for LM.

    Returns:
        The fit result of each optimizer, by optimizer name.
    """
    lower = parameters.lower_vector(specs)
    upper = parameters.upper_vector(specs)
    initial = parameters.initial_vector(specs)
    results = {}
    if variant.family == models.YEOH:
        results[LINEAR] = fitting.fit_linear_least_squares(
            residual, lower, upper
        )
    results[LM] = fitting.fit_levenberg_marquardt(residual, initial, jacobian)
    results[CMA_ES] = fitting.fit_cma_es(
        residual, initial, parameters.scale_vector(specs), lower, upper
    )
    return results


def fit_homogeneous(
    variants: list[models.ModelVariant],
    tests: list[data.HomogeneousTest],
    source_key: str,
) -> dict[str, dict[str, fitting.FitResult]]:
    """Fits every variant to homogeneous tests with every optimizer.

    Args:
        variants: Incompressible model variants.
        tests: The training tests.
        source_key: The key of the data source.

    Returns:
        The fit results, by variant name and then by optimizer name.
    """
    results = {}
    for variant in variants:
        residual = fitting.homogeneous_residual_function(variant, tests)
        specs = parameters.parameter_specs(variant, source_key)
        results[variant.name] = fit_with_all_optimizers(
            variant, residual, specs
        )
    return results


def fit_field(
    variants: list[models.ModelVariant],
    steps: list[data.FieldStep],
    source_key: str,
    balance: float,
) -> dict[str, dict[str, fitting.FitResult]]:
    """Fits every variant to field steps through the weak-form residual.

    Args:
        variants: Compressible model variants.
        steps: The training load steps.
        source_key: The key of the data source.
        balance: The weight of the reaction force part of the residual.

    Returns:
        The fit results, by variant name and then by optimizer name.
    """
    results = {}
    for variant in variants:
        residual = fitting.field_residual_function(variant, steps, balance)
        jacobian = fitting.field_jacobian_function(variant, steps, balance)
        specs = parameters.parameter_specs(variant, source_key)
        results[variant.name] = fit_with_all_optimizers(
            variant, residual, specs, jacobian
        )
    return results


def best_result(results: dict[str, fitting.FitResult]) -> fitting.FitResult:
    """Selects the fit result with the lowest cost.

    Args:
        results: Fit results, by optimizer name.

    Returns:
        The result with the lowest cost.
    """
    return min(results.values(), key=lambda result: result.cost)


def optimizer_table(
    variants: list[models.ModelVariant],
    results: dict[str, dict[str, fitting.FitResult]],
) -> pd.DataFrame:
    """Tabulates the cost and the effort of every optimizer.

    Args:
        variants: The model variants.
        results: The fit results, by variant name and optimizer name.

    Returns:
        A table with one row per variant and optimizer.
    """
    rows = []
    for variant in variants:
        for method, result in results[variant.name].items():
            row = {
                "variant": variant.title,
                "optimizer": method,
                "cost": result.cost,
                "evaluations": result.n_evaluations,
            }
            row.update(
                dict(zip(variant.parameter_names, result.theta, strict=True))
            )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["variant", "optimizer"])


def expected_table(
    variants: list[models.ModelVariant],
    results: dict[str, dict[str, fitting.FitResult]],
    source_key: str,
) -> pd.DataFrame:
    """Compares the best fitted parameters with the expected values.

    Args:
        variants: The model variants.
        results: The fit results, by variant name and optimizer name.
        source_key: The key of the data source.

    Returns:
        A table with one row per variant and parameter.
    """
    tables = []
    for variant in variants:
        specs = parameters.parameter_specs(variant, source_key)
        theta = best_result(results[variant.name]).theta
        table = evaluation.expected_comparison(specs, theta)
        table.insert(0, "variant", variant.title)
        tables.append(table)
    return (
        pd.concat(tables)
        .rename_axis("parameter")
        .set_index("variant", append=True)
    )


def stress_metrics_table(
    variants: list[models.ModelVariant],
    results: dict[str, dict[str, fitting.FitResult]],
    tests: list[data.HomogeneousTest],
    train_modes: tuple[str, ...],
) -> pd.DataFrame:
    """Computes the error metrics of every variant on every test.

    Args:
        variants: Incompressible model variants.
        results: The fit results, by variant name and optimizer name.
        tests: The tests to evaluate.
        train_modes: The test modes that the fit used.

    Returns:
        A table with one row per variant and test.
    """
    rows = []
    for variant in variants:
        theta = best_result(results[variant.name]).theta
        for test in tests:
            row = evaluation.stress_metrics(variant, theta, test)
            row["role"] = "train" if test.mode in train_modes else "test"
            rows.append(row)
    return pd.DataFrame(rows).set_index(["variant", "mode"])


def stress_curves(
    variants: list[models.ModelVariant],
    results: dict[str, dict[str, fitting.FitResult]],
    mode: str,
    stretch: np.ndarray,
) -> dict[str, np.ndarray]:
    """Computes the model stress curve of every variant on a stretch grid.

    Args:
        variants: Incompressible model variants.
        results: The fit results, by variant name and optimizer name.
        mode: The test mode.
        stretch: The stretch grid.

    Returns:
        The stress curve of each variant, by variant title.
    """
    curves = {}
    for variant in variants:
        theta = best_result(results[variant.name]).theta
        test = data.HomogeneousTest(
            "grid", "grid", mode, stretch, np.ones_like(stretch)
        )
        curves[variant.title] = evaluation.predict_stress(variant, theta, test)
    return curves


def relative_residuals(
    variants: list[models.ModelVariant],
    results: dict[str, dict[str, fitting.FitResult]],
    test: data.HomogeneousTest,
) -> dict[str, np.ndarray]:
    """Computes the relative stress residual of every variant on a test.

    Args:
        variants: Incompressible model variants.
        results: The fit results, by variant name and optimizer name.
        test: The homogeneous test.

    Returns:
        The relative residual at each point, by variant title.
    """
    residuals = {}
    for variant in variants:
        theta = best_result(results[variant.name]).theta
        model = evaluation.predict_stress(variant, theta, test)
        residuals[variant.title] = (
            model - test.nominal_stress
        ) / test.nominal_stress
    return residuals


def stability_table(
    variants: list[models.ModelVariant],
    results: dict[str, dict[str, fitting.FitResult]],
    max_first_invariant: float,
) -> pd.DataFrame:
    """Checks dW/dI1 > 0 for every Yeoh fit over a strain range.

    Args:
        variants: Model variants. The function skips the non-Yeoh variants.
        results: The fit results, by variant name and optimizer name.
        max_first_invariant: The upper end of the I1 range.

    Returns:
        A table with the stability margin of each Yeoh variant.
    """
    rows = []
    for variant in variants:
        if variant.family != models.YEOH:
            continue
        theta = best_result(results[variant.name]).theta
        coefficients = theta[:-1] if variant.compressible else theta
        margin = evaluation.yeoh_stability_margin(
            coefficients, max_first_invariant
        )
        rows.append(
            {
                "variant": variant.title,
                "min dW/dI1": margin,
                "stable": margin > 0,
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def solve_field_steps(
    variant: models.ModelVariant, theta: np.ndarray, steps: list[data.FieldStep]
) -> list[fem.EquilibriumResult]:
    """Solves the forward problem at several load steps.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector.
        steps: The field steps.

    Returns:
        One equilibrium result per load step.
    """
    return [fem.solve_equilibrium(variant, theta, step) for step in steps]


def field_metrics_table(
    variant: models.ModelVariant,
    theta: np.ndarray,
    steps: list[data.FieldStep],
    solutions: list[fem.EquilibriumResult],
    train_steps: tuple[int, ...],
) -> pd.DataFrame:
    """Computes the field and reaction errors of forward solutions.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector.
        steps: The field steps with the measured data.
        solutions: The forward solutions, one per step.
        train_steps: The load steps that the fit used.

    Returns:
        A table with one row per load step.
    """
    rows = []
    for step, solution in zip(steps, solutions, strict=True):
        reactions = fem.predicted_reactions(
            variant, theta, solution.displacement, step
        )
        rows.append(
            {
                "variant": variant.title,
                "load step": step.load_step,
                "role": "train" if step.load_step in train_steps else "test",
                "displacement error": evaluation.field_displacement_error(
                    solution.displacement, step
                ),
                "reaction error": evaluation.reaction_relative_error(
                    reactions, step.reactions
                ),
                "solver iterations": solution.iterations,
                "converged": solution.success,
            }
        )
    return pd.DataFrame(rows).set_index(["variant", "load step"])


def verification_table(
    variant: models.ModelVariant, theta: np.ndarray, steps: list[data.FieldStep]
) -> pd.DataFrame:
    """Compares the model with the stresses and forces of the source solver.

    With the true parameters, the model must reproduce the published
    stresses, internal forces and reaction forces.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector.
        steps: The field steps.

    Returns:
        The largest absolute difference of each quantity, per load step.
    """
    rows = []
    theta_tensor = kinematics.as_tensor(theta)
    for step in steps:
        mesh = fem.mesh_from_step(step)
        displacement = kinematics.as_tensor(step.displacement)
        gradient = fem.deformation_gradient(displacement, mesh)
        stress = mechanics.first_piola_kirchhoff(
            variant, gradient, theta_tensor
        )
        forces = fem.internal_forces(variant, theta_tensor, displacement, mesh)
        reactions = fem.reaction_forces(forces, mesh).detach().numpy()
        rows.append(
            {
                "load step": step.load_step,
                "max |P - P_source|": float(
                    np.max(
                        np.abs(
                            stress.detach().numpy()[:, :2, :2]
                            - step.reference_stress
                        )
                    )
                ),
                "max |f - f_source|": float(
                    np.max(
                        np.abs(
                            forces.detach().numpy()
                            - step.reference_internal_forces
                        )
                    )
                ),
                "max |R - R_source|": float(
                    np.max(np.abs(reactions - step.reactions))
                ),
                "max |P_source|": float(np.max(np.abs(step.reference_stress))),
            }
        )
    return pd.DataFrame(rows).set_index("load step")


def normalized_uniaxial_curve(
    variant: models.ModelVariant, theta: np.ndarray, stretch: np.ndarray
) -> np.ndarray:
    """Computes the uniaxial stress divided by the initial shear modulus.

    The volumetric parameter is not used: the uniaxial curve assumes an
    incompressible material.

    Args:
        variant: A Yeoh model variant.
        theta: The parameter vector of the variant.
        stretch: The stretch grid.

    Returns:
        P / mu0 at each stretch.
    """
    order = len(
        [name for name in variant.parameter_names if name.startswith("C")]
    )
    incompressible = models.yeoh_variant(order)
    coefficients = theta[:order]
    test = data.HomogeneousTest(
        "grid", "grid", data.UNIAXIAL, stretch, np.ones_like(stretch)
    )
    stress = evaluation.predict_stress(incompressible, coefficients, test)
    return stress / evaluation.initial_shear_modulus(
        incompressible, coefficients
    )


def variability_table(
    fitted: dict[str, np.ndarray], units: dict[str, str]
) -> pd.DataFrame:
    """Compares the Yeoh N=3 parameters across data sources.

    The ratios C20 / C10 and C30 / C10 describe the shape of the response.
    They do not depend on the stiffness of the material.

    Args:
        fitted: The fitted Yeoh N=3 vector of each source, by source key.
            The first three entries are C10, C20 and C30.
        units: The stress unit of each source, by source key.

    Returns:
        A table with one row per data source.
    """
    rows = []
    for source_key, theta in fitted.items():
        c10, c20, c30 = theta[:3]
        rows.append(
            {
                "source": source_key,
                "unit": units[source_key],
                "C10": c10,
                "C20": c20,
                "C30": c30,
                "mu0 = 2 C10": 2.0 * c10,
                "C20 / C10": c20 / c10,
                "C30 / C10": c30 / c10,
            }
        )
    return pd.DataFrame(rows).set_index("source")


def solve_field_variants(
    variants: list[models.ModelVariant],
    results: dict[str, dict[str, fitting.FitResult]],
    steps: list[data.FieldStep],
    train_steps: tuple[int, ...],
) -> tuple[dict[str, list[fem.EquilibriumResult]], pd.DataFrame]:
    """Solves the forward problem for every fitted field variant.

    Args:
        variants: Compressible model variants.
        results: The fit results, by variant name and optimizer name.
        steps: The field steps.
        train_steps: The load steps that the fit used.

    Returns:
        The solutions of each variant, by variant name, and a table of the
        field and reaction errors.
    """
    solutions = {}
    tables = []
    for variant in variants:
        theta = best_result(results[variant.name]).theta
        solutions[variant.name] = solve_field_steps(variant, theta, steps)
        tables.append(
            field_metrics_table(
                variant, theta, steps, solutions[variant.name], train_steps
            )
        )
    return solutions, pd.concat(tables)


def reaction_predictions(
    variants: list[models.ModelVariant],
    results: dict[str, dict[str, fitting.FitResult]],
    steps: list[data.FieldStep],
    solutions: dict[str, list[fem.EquilibriumResult]],
    group: int,
) -> dict[str, list[float]]:
    """Collects the predicted reaction force of one group over the steps.

    Args:
        variants: Compressible model variants.
        results: The fit results, by variant name and optimizer name.
        steps: The field steps.
        solutions: The forward solutions, by variant name.
        group: The boundary code of the reaction force group.

    Returns:
        The reaction force at each load step, by variant title.
    """
    predictions = {}
    for variant in variants:
        theta = best_result(results[variant.name]).theta
        predictions[variant.title] = [
            float(
                fem.predicted_reactions(
                    variant, theta, solution.displacement, step
                )[group - 1]
            )
            for step, solution in zip(
                steps, solutions[variant.name], strict=True
            )
        ]
    return predictions


def normalized_curves(
    fits: dict[str, tuple[models.ModelVariant, np.ndarray]], stretch: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Computes P / mu0 in uniaxial tension for several Yeoh fits.

    Args:
        fits: The variant and the fitted vector, by legend label.
        stretch: The stretch grid.

    Returns:
        (stretch, P / mu0) of each fit, by legend label.
    """
    return {
        label: (stretch, normalized_uniaxial_curve(variant, theta, stretch))
        for label, (variant, theta) in fits.items()
    }


def tangent_check(
    variant: models.ModelVariant,
    theta: np.ndarray,
    deformation_gradient: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Checks the rank-4 tangent of a Yeoh N=1 + volumetric fit.

    Args:
        variant: The Yeoh N=1 + volumetric variant.
        theta: The fitted vector [C10, D1].
        deformation_gradient: A finite deformation gradient, shape (3, 3).

    Returns:
        The largest difference between the tangent at F = I and the linear
        elasticity tensor, and a table of the index symmetries at F = I and
        at the finite deformation.
    """
    theta_tensor = kinematics.as_tensor(theta)
    at_identity = mechanics.material_tangent(
        variant, kinematics.as_tensor(np.eye(3)), theta_tensor
    ).numpy()
    at_deformation = mechanics.material_tangent(
        variant, kinematics.as_tensor(deformation_gradient), theta_tensor
    ).numpy()
    linear = evaluation.isotropic_linear_tangent(
        2.0 * theta[0], 2.0 * theta[-1]
    )
    difference = float(np.abs(at_identity - linear).max())
    table = pd.DataFrame(
        [
            evaluation.tangent_symmetries(at_identity),
            evaluation.tangent_symmetries(at_deformation),
        ],
        index=["F = I", "finite F"],
    )
    return difference, table
