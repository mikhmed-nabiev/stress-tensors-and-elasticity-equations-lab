"""Stresses and tangents from a strain energy, by automatic differentiation.

The first Piola-Kirchhoff stress P = dW/dF is a rank-2 tensor. In a
homogeneous test, P11 is the nominal stress. The material tangent
A = dP/dF = d^2 W / dF dF is a rank-4 tensor with 3^4 = 81 components.

torch autograd computes both derivatives from the compiled DEAP tree. So
no stress formula is written by hand.
"""

import torch

from hyperfit import data, kinematics, models

# In an equibiaxial test, W depends on the stretch through two equal
# principal stretches. dW/dlambda then counts the stress twice.
_MODE_FACTOR = {data.UNIAXIAL: 1.0, data.EQUIBIAXIAL: 0.5, data.PURE_SHEAR: 1.0}


def _differentiable(values: torch.Tensor) -> torch.Tensor:
    """Makes a tensor a valid input for torch.autograd.grad.

    A tensor that is already part of a torch graph stays unchanged. So
    higher derivatives through the tensor stay possible.

    Args:
        values: Any tensor.

    Returns:
        A tensor that requires a gradient.
    """
    if values.requires_grad:
        return values
    return values.clone().requires_grad_(True)


def nominal_stress(
    variant: models.ModelVariant,
    mode: str,
    stretch: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Computes the nominal stress of a homogeneous test.

    The function forms the reduced energy W(lambda) for the test mode and
    differentiates it with respect to the stretch lambda. The torch graph
    stays open, so a caller can differentiate the result with respect to
    the parameter vector.

    Args:
        variant: An incompressible model variant.
        mode: The test mode. One of `data.TEST_MODES`.
        stretch: The stretch in the loading direction, shape (n,).
        theta: The parameter vector, shape (n_parameters,).

    Returns:
        The nominal stress in the loading direction, shape (n,).
    """
    stretch = _differentiable(stretch)
    principal = kinematics.mode_stretches(mode, stretch)
    inputs = kinematics.inputs_from_stretches(principal)
    energy = models.strain_energy(variant, inputs, theta)
    (derivative,) = torch.autograd.grad(
        energy.sum(), stretch, create_graph=True
    )
    return _MODE_FACTOR[mode] * derivative


def first_piola_kirchhoff(
    variant: models.ModelVariant,
    deformation_gradient: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Computes the first Piola-Kirchhoff stress P = dW/dF.

    Args:
        variant: A model variant with invariant inputs.
        deformation_gradient: F, shape (n, 3, 3).
        theta: The parameter vector, shape (n_parameters,).

    Returns:
        P, shape (n, 3, 3).
    """
    deformation_gradient = _differentiable(deformation_gradient)
    inputs = kinematics.inputs_from_deformation(deformation_gradient)
    energy = models.strain_energy(variant, inputs, theta)
    (stress,) = torch.autograd.grad(
        energy.sum(), deformation_gradient, create_graph=True
    )
    return stress


def material_tangent(
    variant: models.ModelVariant,
    deformation_gradient: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Computes the rank-4 material tangent A_iJkL = dP_iJ / dF_kL.

    Args:
        variant: A model variant with invariant inputs.
        deformation_gradient: F at one material point, shape (3, 3).
        theta: The parameter vector, shape (n_parameters,).

    Returns:
        A, shape (3, 3, 3, 3).
    """

    def stress_at_point(gradient: torch.Tensor) -> torch.Tensor:
        return first_piola_kirchhoff(variant, gradient[None], theta)[0]

    return torch.autograd.functional.jacobian(
        stress_at_point, deformation_gradient
    )
