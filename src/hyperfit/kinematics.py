"""Kinematics in three dimensions, written with torch tensors.

Notation:

* F is the deformation gradient. F maps a reference line element to the
  deformed line element.
* C = F^T F is the right Cauchy-Green tensor.
* J = det F is the volume ratio.
* I1 = tr C is the first invariant.
* I1b = J^(-2/3) I1 is the isochoric first invariant. It does not change
  under a pure volume change.

All functions keep the torch graph, so autograd can differentiate them.
"""

import numpy as np
import torch

from hyperfit import data

DTYPE = torch.float64


def as_tensor(values: np.ndarray | float) -> torch.Tensor:
    """Converts numbers to a float64 torch tensor.

    Args:
        values: A number or an array.

    Returns:
        A new float64 tensor with a copy of the values.
    """
    return torch.tensor(np.asarray(values, dtype=np.float64), dtype=DTYPE)


def mode_stretches(mode: str, stretch: torch.Tensor) -> torch.Tensor:
    """Computes the three principal stretches of a homogeneous test.

    All modes assume an incompressible material, so the product of the
    three stretches is 1.

    Args:
        mode: The test mode. One of `data.TEST_MODES`.
        stretch: The stretch in the loading direction, shape (n,).

    Returns:
        The principal stretches, shape (n, 3).

    Raises:
        ValueError: The mode is unknown.
    """
    if mode == data.UNIAXIAL:
        lateral = stretch ** (-0.5)
        return torch.stack([stretch, lateral, lateral], dim=-1)
    if mode == data.EQUIBIAXIAL:
        return torch.stack([stretch, stretch, stretch ** (-2.0)], dim=-1)
    if mode == data.PURE_SHEAR:
        return torch.stack(
            [stretch, torch.ones_like(stretch), 1.0 / stretch], dim=-1
        )
    raise ValueError(f"Unknown test mode: {mode}.")


def inputs_from_stretches(principal: torch.Tensor) -> dict[str, torch.Tensor]:
    """Computes the strain energy inputs from principal stretches.

    Args:
        principal: The principal stretches, shape (n, 3).

    Returns:
        A mapping with the isochoric first invariant "I1b", the volume
        ratio "J" and the principal stretches "l1", "l2" and "l3".
    """
    volume_ratio = principal.prod(dim=-1)
    first_invariant = (principal**2).sum(dim=-1)
    return {
        "I1b": volume_ratio ** (-2.0 / 3.0) * first_invariant,
        "J": volume_ratio,
        "l1": principal[..., 0],
        "l2": principal[..., 1],
        "l3": principal[..., 2],
    }


def element_deformation_gradient(
    nodal_displacement: torch.Tensor,
    connectivity: torch.Tensor,
    shape_gradients: torch.Tensor,
) -> torch.Tensor:
    """Computes the in-plane deformation gradient of linear triangles.

    For a linear triangle, F = I + sum_a u_a (grad N_a)^T. The value is
    constant inside each element.

    Args:
        nodal_displacement: Nodal displacements, shape (n_nodes, 2).
        connectivity: Node indices of each element, shape (n_elements, 3).
        shape_gradients: Shape function gradients, shape (n_elements, 3, 2).

    Returns:
        The in-plane deformation gradient, shape (n_elements, 2, 2).
    """
    element_displacement = nodal_displacement[connectivity]  # (E, 3, 2)
    outer_products = (
        element_displacement[..., :, None] * shape_gradients[..., None, :]
    )
    displacement_gradient = outer_products.sum(dim=1)  # (E, 2, 2)
    identity = torch.eye(2, dtype=DTYPE)
    return identity + displacement_gradient


def plane_strain_deformation_gradient(in_plane: torch.Tensor) -> torch.Tensor:
    """Embeds an in-plane deformation gradient into three dimensions.

    Plane strain fixes the thickness, so F33 = 1 and F13 = F23 = 0.

    Args:
        in_plane: The in-plane deformation gradient, shape (n, 2, 2).

    Returns:
        The full deformation gradient, shape (n, 3, 3).
    """
    padded = torch.nn.functional.pad(in_plane, (0, 1, 0, 1))
    thickness_direction = torch.zeros(3, 3, dtype=DTYPE)
    thickness_direction[2, 2] = 1.0
    return padded + thickness_direction


def right_cauchy_green(deformation_gradient: torch.Tensor) -> torch.Tensor:
    """Computes the right Cauchy-Green tensor C = F^T F.

    Args:
        deformation_gradient: F, shape (n, 3, 3).

    Returns:
        C, shape (n, 3, 3).
    """
    return deformation_gradient.transpose(-1, -2) @ deformation_gradient


def determinant(matrix: torch.Tensor) -> torch.Tensor:
    """Computes the determinant of 3 x 3 matrices by cofactor expansion.

    The explicit formula is faster than an LU factorization for many small
    matrices.

    Args:
        matrix: The matrices, shape (n, 3, 3).

    Returns:
        The determinants, shape (n,).
    """
    m = matrix
    return (
        m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1])
        - m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])
        + m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0])
    )


def inputs_from_deformation(
    deformation_gradient: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Computes the invariant strain energy inputs from F.

    The first invariant uses the identity tr(F^T F) = sum_ij F_ij^2. The
    function does not compute principal stretches, because an eigenvalue
    decomposition has no stable gradient at equal eigenvalues.

    Args:
        deformation_gradient: F, shape (n, 3, 3).

    Returns:
        A mapping with the isochoric first invariant "I1b" and the volume
        ratio "J".
    """
    volume_ratio = determinant(deformation_gradient)
    first_invariant = (deformation_gradient**2).sum(dim=(-2, -1))
    return {
        "I1b": volume_ratio ** (-2.0 / 3.0) * first_invariant,
        "J": volume_ratio,
    }
