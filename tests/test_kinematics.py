"""Tests for hyperfit.kinematics."""

import numpy as np
import pytest
import torch
from conftest import SQUARE_CONNECTIVITY, SQUARE_COORDINATES, triangle_gradients

from hyperfit import data, kinematics


@pytest.mark.parametrize("mode", data.TEST_MODES)
def test_mode_stretches_are_incompressible(mode):
    stretch = kinematics.as_tensor([1.0, 1.5, 3.0])
    principal = kinematics.mode_stretches(mode, stretch)
    assert principal.shape == (3, 3)
    np.testing.assert_allclose(principal.prod(dim=-1).numpy(), 1.0)
    np.testing.assert_allclose(principal[:, 0].numpy(), stretch.numpy())


def test_mode_stretches_rejects_unknown_mode():
    with pytest.raises(ValueError):
        kinematics.mode_stretches("torsion", kinematics.as_tensor([1.0]))


def test_inputs_from_stretches_at_reference_state():
    inputs = kinematics.inputs_from_stretches(
        kinematics.as_tensor([[1.0, 1.0, 1.0]])
    )
    assert inputs["I1b"].item() == pytest.approx(3.0)
    assert inputs["J"].item() == pytest.approx(1.0)


def test_isochoric_invariant_ignores_volume_change():
    principal = kinematics.as_tensor([[1.2, 1.2, 1.2]])
    inputs = kinematics.inputs_from_stretches(principal)
    assert inputs["I1b"].item() == pytest.approx(3.0)
    assert inputs["J"].item() == pytest.approx(1.2**3)


def test_determinant_matches_torch():
    matrices = torch.randn(
        10,
        3,
        3,
        dtype=kinematics.DTYPE,
        generator=torch.Generator().manual_seed(0),
    )
    np.testing.assert_allclose(
        kinematics.determinant(matrices).numpy(),
        torch.linalg.det(matrices).numpy(),
    )


def test_inputs_from_deformation_match_stretches():
    deformation = torch.diag(kinematics.as_tensor([1.3, 0.9, 1.1]))[None]
    from_deformation = kinematics.inputs_from_deformation(deformation)
    from_stretches = kinematics.inputs_from_stretches(
        kinematics.as_tensor([[1.3, 0.9, 1.1]])
    )
    for name in ("I1b", "J"):
        assert from_deformation[name].item() == pytest.approx(
            from_stretches[name].item()
        )


def test_element_deformation_gradient_of_affine_field():
    gradient = np.array([[0.2, 0.1], [-0.05, 0.3]])
    displacement = SQUARE_COORDINATES @ gradient.T
    shape_gradients, _ = triangle_gradients(
        SQUARE_COORDINATES, SQUARE_CONNECTIVITY
    )
    deformation = kinematics.element_deformation_gradient(
        kinematics.as_tensor(displacement),
        torch.as_tensor(SQUARE_CONNECTIVITY),
        kinematics.as_tensor(shape_gradients),
    )
    expected = np.eye(2) + gradient
    for element in deformation.numpy():
        np.testing.assert_allclose(element, expected, atol=1e-12)


def test_plane_strain_embedding():
    in_plane = kinematics.as_tensor([[[1.1, 0.2], [0.0, 0.9]]])
    full = kinematics.plane_strain_deformation_gradient(in_plane)
    assert full.shape == (1, 3, 3)
    assert full[0, 2, 2].item() == 1.0
    assert full[0, :2, 2].abs().sum().item() == 0.0
    np.testing.assert_allclose(full[0, :2, :2].numpy(), in_plane[0].numpy())
