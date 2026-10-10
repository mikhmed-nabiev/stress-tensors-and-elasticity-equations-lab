"""Tests for hyperfit.fitting with synthetic data and known parameters."""

import dataclasses

import numpy as np
import pytest
from conftest import affine_displacement, make_square_step

from hyperfit import (
    data,
    evaluation,
    fem,
    fitting,
    kinematics,
    mechanics,
    models,
)


def synthetic_test(variant, theta, mode, stretch):
    """Builds a noise-free homogeneous test from known parameters."""
    stress = (
        mechanics.nominal_stress(
            variant,
            mode,
            kinematics.as_tensor(stretch),
            kinematics.as_tensor(theta),
        )
        .detach()
        .numpy()
    )
    return data.HomogeneousTest("synthetic", "synthetic", mode, stretch, stress)


YEOH = models.yeoh_variant(3)
YEOH_THETA = np.array([0.2, -0.002, 0.00005])
STRETCH = np.linspace(1.1, 5.0, 25)


@pytest.fixture
def yeoh_residual():
    test = synthetic_test(YEOH, YEOH_THETA, data.UNIAXIAL, STRETCH)
    return fitting.homogeneous_residual_function(YEOH, [test])


def test_residual_is_zero_at_true_parameters(yeoh_residual):
    assert (
        np.abs(fitting.evaluate_residual(yeoh_residual, YEOH_THETA)).max()
        < 1e-12
    )


def test_linearize_reproduces_linear_residual(yeoh_residual):
    matrix, vector = fitting.linearize(yeoh_residual, 3)
    assert fitting.is_linear(
        yeoh_residual, matrix, vector, np.array([0.3, 0.1, -0.2])
    )


def test_ogden_residual_is_not_linear():
    variant = models.ogden_variant(2)
    test = synthetic_test(
        variant, [0.6, 1.3, 0.001, 5.0], data.UNIAXIAL, STRETCH
    )
    residual = fitting.homogeneous_residual_function(variant, [test])
    matrix, vector = fitting.linearize(residual, 4)
    assert not fitting.is_linear(
        residual, matrix, vector, np.array([0.5, 1.5, 0.01, 4.0])
    )


def test_linear_least_squares_recovers_parameters(yeoh_residual):
    result = fitting.fit_linear_least_squares(
        yeoh_residual, np.array([0.0, -1.0, -1.0]), np.array([10.0, 1.0, 1.0])
    )
    np.testing.assert_allclose(result.theta, YEOH_THETA, rtol=1e-8)


def test_levenberg_marquardt_recovers_parameters(yeoh_residual):
    result = fitting.fit_levenberg_marquardt(
        yeoh_residual, np.array([0.1, 0.0, 0.0])
    )
    np.testing.assert_allclose(result.theta, YEOH_THETA, rtol=1e-6)


def test_cma_es_recovers_parameters(yeoh_residual):
    result = fitting.fit_cma_es(
        yeoh_residual,
        initial=np.array([0.1, 0.0, 0.0]),
        scale=np.array([0.1, 0.01, 0.001]),
        lower=np.array([0.0, -1.0, -1.0]),
        upper=np.array([10.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(result.theta, YEOH_THETA, rtol=1e-4)
    assert len(result.history) > 0
    assert result.history[-1] <= result.history[0]


def test_ogden_fit_with_levenberg_marquardt():
    variant = models.ogden_variant(2)
    theta = np.array([0.6, 1.3, 0.001, 5.0])
    tests = [
        synthetic_test(variant, theta, mode, STRETCH[:15])
        for mode in data.TEST_MODES
    ]
    residual = fitting.homogeneous_residual_function(variant, tests)
    result = fitting.fit_levenberg_marquardt(
        residual, np.array([0.5, 1.5, 0.002, 4.5])
    )
    model = evaluation.predict_stress(variant, result.theta, tests[0])
    np.testing.assert_allclose(model, tests[0].nominal_stress, rtol=1e-6)


def test_field_fit_recovers_parameters():
    variant = models.yeoh_variant(1, compressible=True)
    theta = np.array([0.5, 1.5])
    displacement = affine_displacement(np.array([[0.1, 0.02], [0.0, -0.03]]))
    step = make_square_step(displacement, np.zeros(4))
    reactions = fem.predicted_reactions(variant, theta, displacement, step)
    step = dataclasses.replace(step, reactions=reactions)
    residual = fitting.field_residual_function(variant, [step], 100.0)
    jacobian = fitting.field_jacobian_function(variant, [step], 100.0)
    linear = fitting.fit_linear_least_squares(
        residual, np.zeros(2), np.full(2, 10.0)
    )
    levenberg = fitting.fit_levenberg_marquardt(
        residual, np.array([1.0, 1.0]), jacobian
    )
    np.testing.assert_allclose(linear.theta, theta, rtol=1e-8)
    np.testing.assert_allclose(levenberg.theta, theta, rtol=1e-8)
