"""Tests for hyperfit.fem on the small unit square mesh."""

import dataclasses

import numpy as np
import pytest
from conftest import affine_displacement, make_square_step

from hyperfit import fem, fitting, kinematics, models

THETA = np.array([0.5, 1.5])
VARIANT = models.yeoh_variant(1, compressible=True)


def test_energy_is_zero_without_deformation(square_step):
    mesh = fem.mesh_from_step(square_step)
    zero = kinematics.as_tensor(np.zeros((5, 2)))
    energy = fem.total_energy(VARIANT, kinematics.as_tensor(THETA), zero, mesh)
    assert abs(energy.item()) < 1e-14


def test_homogeneous_field_has_no_interior_force(square_step):
    mesh = fem.mesh_from_step(square_step)
    forces = fem.internal_forces(
        VARIANT,
        kinematics.as_tensor(THETA),
        kinematics.as_tensor(square_step.displacement),
        mesh,
    )
    assert np.abs(forces.detach().numpy()[4]).max() < 1e-12


def test_reaction_groups_balance(square_step):
    reactions = fem.predicted_reactions(
        VARIANT, THETA, square_step.displacement, square_step
    )
    assert reactions.shape == (4,)
    assert reactions[0] == pytest.approx(-reactions[1])
    assert reactions[2] == pytest.approx(-reactions[3])
    assert reactions[1] > 0.0


def test_weak_form_residual_is_zero_for_consistent_data(square_step):
    reactions = fem.predicted_reactions(
        VARIANT, THETA, square_step.displacement, square_step
    )
    step = dataclasses.replace(square_step, reactions=reactions)
    mesh = fem.mesh_from_step(step)
    residual = fem.weak_form_residual(
        VARIANT, kinematics.as_tensor(THETA), step, mesh, 100.0
    )
    assert np.abs(residual.detach().numpy()).max() < 1e-12


def test_weak_form_jacobian_matches_linearization(square_step):
    variant = models.yeoh_variant(3, compressible=True)
    theta = np.array([0.5, 0.1, 0.01, 1.5])
    mesh = fem.mesh_from_step(square_step)
    jacobian = fem.weak_form_jacobian(
        variant, kinematics.as_tensor(theta), square_step, mesh, 4.0
    )
    residual = fitting.field_residual_function(variant, [square_step], 4.0)
    matrix, _ = fitting.linearize(residual, 4)
    np.testing.assert_allclose(jacobian.detach().numpy(), matrix, atol=1e-12)


def test_solver_recovers_affine_interior_displacement():
    gradient = np.array([[0.2, 0.0], [0.0, -0.05]])
    exact = affine_displacement(gradient)
    start = exact.copy()
    start[4] = [0.0, 0.0]
    step = make_square_step(exact, np.zeros(4))
    result = fem.solve_equilibrium(
        VARIANT, THETA, step, initial_displacement=start
    )
    assert result.success
    np.testing.assert_allclose(result.displacement, exact, atol=1e-7)


def test_homogeneous_initial_guess_keeps_boundary_values(square_step):
    guess = fem.homogeneous_initial_guess(square_step)
    prescribed = square_step.dirichlet_codes != 0
    np.testing.assert_allclose(
        guess[prescribed], square_step.displacement[prescribed]
    )
    np.testing.assert_allclose(guess[4], [0.05, 0.025])
