"""Error metrics, physical checks and result tables."""

import numpy as np
import pandas as pd

from hyperfit import data, kinematics, mechanics, models, parameters, trees


def relative_rmse(model: np.ndarray, measured: np.ndarray) -> float:
    """Computes the root mean square of the relative error.

    Args:
        model: Model values.
        measured: Measured values. No value may be zero.

    Returns:
        sqrt(mean(((model - measured) / measured)^2)).
    """
    relative = (model - measured) / measured
    return float(np.sqrt(np.mean(relative**2)))


def r_squared(model: np.ndarray, measured: np.ndarray) -> float:
    """Computes the coefficient of determination R^2.

    Args:
        model: Model values.
        measured: Measured values.

    Returns:
        1 - (residual sum of squares) / (total sum of squares).
    """
    residual = float(np.sum((measured - model) ** 2))
    total = float(np.sum((measured - measured.mean()) ** 2))
    return 1.0 - residual / total


def predict_stress(
    variant: models.ModelVariant, theta: np.ndarray, test: data.HomogeneousTest
) -> np.ndarray:
    """Computes the model nominal stress at the stretches of a test.

    Args:
        variant: An incompressible model variant.
        theta: The parameter vector.
        test: The homogeneous test.

    Returns:
        The model nominal stress at each point of the test.
    """
    stress = mechanics.nominal_stress(
        variant,
        test.mode,
        kinematics.as_tensor(test.stretch),
        kinematics.as_tensor(theta),
    )
    return stress.detach().numpy()


def stress_metrics(
    variant: models.ModelVariant, theta: np.ndarray, test: data.HomogeneousTest
) -> dict[str, float | str | int]:
    """Computes the error metrics of one variant on one test.

    Args:
        variant: An incompressible model variant.
        theta: The parameter vector.
        test: The homogeneous test.

    Returns:
        A row with the relative RMSE, R^2 and the number of points.
    """
    model = predict_stress(variant, theta, test)
    return {
        "source": test.source_key,
        "mode": test.mode,
        "variant": variant.title,
        "relative_rmse": relative_rmse(model, test.nominal_stress),
        "r_squared": r_squared(model, test.nominal_stress),
        "n_points": int(test.stretch.size),
    }


def initial_shear_modulus(
    variant: models.ModelVariant, theta: np.ndarray
) -> float:
    """Computes the shear modulus mu0 at zero strain.

    For Yeoh, mu0 = 2 C10. For Ogden, mu0 = 0.5 sum_p mu_p alpha_p.

    Args:
        variant: The model variant.
        theta: The parameter vector.

    Returns:
        mu0, in the stress unit of the parameters.
    """
    if variant.family == models.YEOH:
        return 2.0 * float(theta[0])
    moduli = theta[0::2]
    exponents = theta[1::2]
    return 0.5 * float(np.sum(moduli * exponents))


def yeoh_stability_margin(
    theta: np.ndarray, max_first_invariant: float
) -> float:
    """Computes the smallest dW/dI1 of a Yeoh model over a strain range.

    The Yeoh model is stable when dW/dI1 > 0 for every I1 in the range.
    A negative margin marks a range where the material would soften
    without limit.

    Args:
        theta: The Yeoh coefficients C10, C20, ... (no volumetric term).
        max_first_invariant: The upper end of the I1 range.

    Returns:
        The minimum of dW/dI1 over I1 in [3, max_first_invariant].
    """
    deviation = np.linspace(0.0, max_first_invariant - 3.0, 400)
    derivative = sum(
        (order + 1) * coefficient * deviation**order
        for order, coefficient in enumerate(theta)
    )
    return float(np.min(derivative))


def max_first_invariant(test: data.HomogeneousTest) -> float:
    """Computes the largest first invariant I1 that a test reaches.

    Args:
        test: The homogeneous test.

    Returns:
        The maximum of I1 = l1^2 + l2^2 + l3^2 over the test points.
    """
    principal = kinematics.mode_stretches(
        test.mode, kinematics.as_tensor(test.stretch)
    )
    return float((principal**2).sum(dim=-1).max())


def field_displacement_error(
    predicted: np.ndarray, step: data.FieldStep
) -> float:
    """Computes the relative displacement error at the free nodes.

    Args:
        predicted: Predicted nodal displacements, shape (n_nodes, 2).
        step: The field step with the measured displacements.

    Returns:
        ||u_predicted - u_measured|| / ||u_measured|| over the free degrees
        of freedom.
    """
    free = step.dirichlet_codes == 0
    difference = predicted[free] - step.displacement[free]
    return float(
        np.linalg.norm(difference) / np.linalg.norm(step.displacement[free])
    )


def reaction_relative_error(
    predicted: np.ndarray, measured: np.ndarray
) -> float:
    """Computes the relative error of the reaction forces.

    Args:
        predicted: Predicted reaction forces.
        measured: Measured reaction forces.

    Returns:
        ||R_predicted - R_measured|| / ||R_measured||.
    """
    return float(
        np.linalg.norm(predicted - measured) / np.linalg.norm(measured)
    )


def expected_comparison(
    specs: list[parameters.ParameterSpec], fitted: np.ndarray
) -> pd.DataFrame:
    """Compares fitted parameters with the expected values.

    Args:
        specs: The parameter specifications.
        fitted: The fitted parameter vector.

    Returns:
        A table with the expected value, the fitted value, their difference,
        their ratio and the reference. The ratio is NaN where the expected
        value is zero or missing.
    """
    expected = parameters.expected_vector(specs)
    safe_expected = np.where(expected == 0.0, np.nan, expected)
    return pd.DataFrame(
        {
            "expected": expected,
            "fitted": fitted,
            "fitted - expected": fitted - expected,
            "fitted / expected": fitted / safe_expected,
            "unit": [spec.unit for spec in specs],
            "reference": [spec.reference for spec in specs],
        },
        index=[spec.name for spec in specs],
    )


def tree_distance_table(variants: list[models.ModelVariant]) -> pd.DataFrame:
    """Computes the tree edit distance between every pair of variants.

    Args:
        variants: The model variants.

    Returns:
        A square table of tree edit distances.
    """
    titles = [variant.title for variant in variants]
    distances = [
        [
            trees.tree_edit_distance(first.tree, second.tree)
            for second in variants
        ]
        for first in variants
    ]
    return pd.DataFrame(distances, index=titles, columns=titles)


def tangent_symmetries(tangent: np.ndarray) -> dict[str, float]:
    """Measures the index symmetries of a rank-4 tangent A_iJkL.

    Major symmetry: A_iJkL = A_kLiJ. It holds for every hyperelastic
    material, because A is a second derivative of W.
    Minor symmetry: A_iJkL = A_JikL. It holds only in the reference state
    without initial stress.

    Args:
        tangent: The tangent, shape (3, 3, 3, 3).

    Returns:
        The largest deviation from each symmetry and the norm of A.
    """
    major = np.abs(tangent - tangent.transpose(2, 3, 0, 1)).max()
    minor = np.abs(tangent - tangent.transpose(1, 0, 2, 3)).max()
    return {
        "components": int(tangent.size),
        "max |A_iJkL - A_kLiJ|": float(major),
        "max |A_iJkL - A_JikL|": float(minor),
        "max |A|": float(np.abs(tangent).max()),
    }


def isotropic_linear_tangent(
    shear_modulus: float, bulk_modulus: float
) -> np.ndarray:
    """Builds the isotropic linear elasticity tensor.

    C_ijkl = K d_ij d_kl + mu (d_ik d_jl + d_il d_jk - 2/3 d_ij d_kl).

    Args:
        shear_modulus: The shear modulus mu.
        bulk_modulus: The bulk modulus K.

    Returns:
        The tensor, shape (3, 3, 3, 3).
    """
    delta = np.eye(3)
    volumetric = np.einsum("ij,kl->ijkl", delta, delta)
    symmetric = np.einsum("ik,jl->ijkl", delta, delta) + np.einsum(
        "il,jk->ijkl", delta, delta
    )
    return bulk_modulus * volumetric + shear_modulus * (
        symmetric - 2.0 / 3.0 * volumetric
    )
