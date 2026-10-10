"""Tests for hyperfit.models and hyperfit.mechanics."""

import numpy as np
import pytest
import torch

from hyperfit import data, kinematics, mechanics, models

YEOH_THETA = [0.2, -0.01, 0.001]


def closed_form_yeoh_stress(
    mode: str, stretch: np.ndarray, theta: list[float]
) -> np.ndarray:
    """Returns the textbook Yeoh nominal stress 2 (l - l^-k) W1."""
    exponents = {data.UNIAXIAL: 2, data.EQUIBIAXIAL: 5, data.PURE_SHEAR: 3}
    principal = kinematics.mode_stretches(
        mode, kinematics.as_tensor(stretch)
    ).numpy()
    deviation = (principal**2).sum(axis=1) - 3.0
    derivative = sum((i + 1) * c * deviation**i for i, c in enumerate(theta))
    return 2.0 * (stretch - stretch ** (-exponents[mode])) * derivative


def test_yeoh_variant_parameter_order():
    assert models.yeoh_variant(3).parameter_names == ("C10", "C20", "C30")
    compressible = models.yeoh_variant(2, compressible=True)
    assert compressible.parameter_names == ("C10", "C20", "D1")
    assert compressible.inputs == ("I1b", "J")


def test_yeoh_variant_rejects_bad_order():
    with pytest.raises(ValueError):
        models.yeoh_variant(4)


def test_yeoh_tree_contains_lower_order_tree():
    assert models.yeoh_expression(2, False) in models.yeoh_expression(3, False)


@pytest.mark.parametrize("mode", data.TEST_MODES)
def test_yeoh_stress_matches_closed_form(mode):
    stretch = np.linspace(1.05, 4.0, 7)
    stress = mechanics.nominal_stress(
        models.yeoh_variant(3),
        mode,
        kinematics.as_tensor(stretch),
        kinematics.as_tensor(YEOH_THETA),
    )
    np.testing.assert_allclose(
        stress.detach().numpy(),
        closed_form_yeoh_stress(mode, stretch, YEOH_THETA),
        rtol=1e-12,
    )


def test_ogden_uniaxial_stress_matches_closed_form():
    theta = [0.63, 1.3, 0.0012, 5.0]
    stretch = np.linspace(1.05, 6.0, 9)
    stress = (
        mechanics.nominal_stress(
            models.ogden_variant(2),
            data.UNIAXIAL,
            kinematics.as_tensor(stretch),
            kinematics.as_tensor(theta),
        )
        .detach()
        .numpy()
    )
    expected = sum(
        theta[2 * p]
        * (
            stretch ** (theta[2 * p + 1] - 1)
            - stretch ** (-theta[2 * p + 1] / 2 - 1)
        )
        for p in range(2)
    )
    np.testing.assert_allclose(stress, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "variant", models.homogeneous_variants(), ids=lambda v: v.name
)
@pytest.mark.parametrize("mode", data.TEST_MODES)
def test_stress_is_zero_at_reference_state(variant, mode):
    theta = kinematics.as_tensor(
        np.linspace(0.1, 0.4, variant.n_parameters) + 1.0
    )
    stress = mechanics.nominal_stress(
        variant, mode, kinematics.as_tensor([1.0]), theta
    )
    assert abs(stress.item()) < 1e-12


def test_small_strain_young_modulus_is_six_c10():
    stretch = kinematics.as_tensor([1.0 + 1e-6])
    stress = mechanics.nominal_stress(
        models.yeoh_variant(3),
        data.UNIAXIAL,
        stretch,
        kinematics.as_tensor(YEOH_THETA),
    )
    assert stress.item() / 1e-6 == pytest.approx(6.0 * YEOH_THETA[0], rel=1e-4)


def test_first_piola_kirchhoff_is_zero_at_identity():
    variant = models.yeoh_variant(1, compressible=True)
    stress = mechanics.first_piola_kirchhoff(
        variant,
        torch.eye(3, dtype=kinematics.DTYPE)[None],
        kinematics.as_tensor([0.5, 1.5]),
    )
    assert stress.abs().max().item() < 1e-12


def test_material_tangent_at_identity_is_linear_elasticity():
    shear_modulus, bulk_modulus = 1.0, 3.0
    theta = kinematics.as_tensor([shear_modulus / 2.0, bulk_modulus / 2.0])
    variant = models.yeoh_variant(1, compressible=True)
    tangent = mechanics.material_tangent(
        variant, torch.eye(3, dtype=kinematics.DTYPE), theta
    ).numpy()
    delta = np.eye(3)
    expected = bulk_modulus * np.einsum(
        "ij,kl->ijkl", delta, delta
    ) + shear_modulus * (
        np.einsum("ik,jl->ijkl", delta, delta)
        + np.einsum("il,jk->ijkl", delta, delta)
        - 2.0 / 3.0 * np.einsum("ij,kl->ijkl", delta, delta)
    )
    np.testing.assert_allclose(tangent, expected, atol=1e-10)


def test_material_tangent_has_major_symmetry():
    variant = models.yeoh_variant(3, compressible=True)
    deformation = kinematics.as_tensor(
        [[1.3, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.05]]
    )
    tangent = mechanics.material_tangent(
        variant, deformation, kinematics.as_tensor([0.5, 0.05, 0.001, 1.5])
    )
    np.testing.assert_allclose(
        tangent.numpy(), tangent.permute(2, 3, 0, 1).numpy(), atol=1e-12
    )
