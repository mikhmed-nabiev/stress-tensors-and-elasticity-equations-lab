"""Finite element tools for the plane strain field test.

The balance of linear momentum without inertia and body forces is

    Div P = 0 in the body,  u = u_D on the Dirichlet boundary.

For a hyperelastic material, this boundary value problem is the minimum
of the total potential energy

    Pi(u) = sum_e W(F_e) w_e,

where F_e is the deformation gradient and w_e the area of element e. The
nodal internal forces are f = dPi/du. Equilibrium means f = 0 at every
free degree of freedom. This module computes Pi with torch, so autograd
gives f without a hand-written assembly.

The weak-form residual uses the measured displacements. It collects the
internal forces at the free degrees of freedom and the mismatch of the
reaction forces. For the true parameters, the residual is zero.
"""

import dataclasses

import numpy as np
import scipy.optimize
import torch

from hyperfit import data, kinematics, models


@dataclasses.dataclass(frozen=True)
class Mesh:
    """Torch tensors of a field step that the solver needs.

    Attributes:
        connectivity: Node indices of each element, shape (n_elements, 3).
        shape_gradients: Shape function gradients, shape (n_elements, 3, 2).
        weights: Element areas, shape (n_elements,).
        dirichlet_codes: Boundary codes, shape (n_nodes, 2). Code 0 marks a
            free degree of freedom.
    """

    connectivity: torch.Tensor
    shape_gradients: torch.Tensor
    weights: torch.Tensor
    dirichlet_codes: torch.Tensor

    @property
    def free_mask(self) -> torch.Tensor:
        """A boolean mask of the free degrees of freedom, shape (n_nodes, 2)."""
        return self.dirichlet_codes == 0

    @property
    def n_reaction_groups(self) -> int:
        """The number of reaction force groups."""
        return int(self.dirichlet_codes.max())


@dataclasses.dataclass(frozen=True)
class EquilibriumResult:
    """The result of a forward finite element solve.

    Attributes:
        displacement: Nodal displacements, shape (n_nodes, 2).
        iterations: The number of L-BFGS-B iterations.
        gradient_norm: The largest absolute internal force at a free degree
            of freedom, at the solution.
        success: True if the optimizer reports convergence.
        message: The optimizer message.
    """

    displacement: np.ndarray
    iterations: int
    gradient_norm: float
    success: bool
    message: str


def mesh_from_step(step: data.FieldStep) -> Mesh:
    """Converts the mesh arrays of a field step into torch tensors.

    Args:
        step: The field step.

    Returns:
        The mesh.
    """
    return Mesh(
        connectivity=torch.as_tensor(step.connectivity, dtype=torch.long),
        shape_gradients=kinematics.as_tensor(step.shape_gradients),
        weights=kinematics.as_tensor(step.weights),
        dirichlet_codes=torch.as_tensor(step.dirichlet_codes, dtype=torch.long),
    )


def deformation_gradient(
    nodal_displacement: torch.Tensor, mesh: Mesh
) -> torch.Tensor:
    """Computes the plane strain deformation gradient of every element.

    Args:
        nodal_displacement: Nodal displacements, shape (n_nodes, 2).
        mesh: The mesh.

    Returns:
        F, shape (n_elements, 3, 3).
    """
    in_plane = kinematics.element_deformation_gradient(
        nodal_displacement, mesh.connectivity, mesh.shape_gradients
    )
    return kinematics.plane_strain_deformation_gradient(in_plane)


def total_energy(
    variant: models.ModelVariant,
    theta: torch.Tensor,
    nodal_displacement: torch.Tensor,
    mesh: Mesh,
) -> torch.Tensor:
    """Computes the total potential energy Pi = sum_e W(F_e) w_e.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector, shape (n_parameters,).
        nodal_displacement: Nodal displacements, shape (n_nodes, 2).
        mesh: The mesh.

    Returns:
        Pi as a scalar tensor.
    """
    gradient = deformation_gradient(nodal_displacement, mesh)
    inputs = kinematics.inputs_from_deformation(gradient)
    energy_density = models.strain_energy(variant, inputs, theta)
    return (energy_density * mesh.weights).sum()


def internal_forces(
    variant: models.ModelVariant,
    theta: torch.Tensor,
    nodal_displacement: torch.Tensor,
    mesh: Mesh,
) -> torch.Tensor:
    """Computes the nodal internal forces f = dPi/du.

    The torch graph stays open, so a caller can differentiate the forces
    with respect to the parameter vector.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector, shape (n_parameters,).
        nodal_displacement: Nodal displacements, shape (n_nodes, 2).
        mesh: The mesh.

    Returns:
        The internal forces, shape (n_nodes, 2).
    """
    displacement = nodal_displacement.detach().clone().requires_grad_(True)
    energy = total_energy(variant, theta, displacement, mesh)
    (forces,) = torch.autograd.grad(energy, displacement, create_graph=True)
    return forces


def reaction_forces(forces: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    """Sums the internal forces of each reaction force group.

    Args:
        forces: Nodal internal forces, shape (n_nodes, 2).
        mesh: The mesh.

    Returns:
        One reaction force for each group, shape (n_groups,). Entry k - 1
        belongs to boundary code k.
    """
    groups = [
        forces[mesh.dirichlet_codes == code].sum()
        for code in range(1, mesh.n_reaction_groups + 1)
    ]
    return torch.stack(groups)


def weak_form_residual(
    variant: models.ModelVariant,
    theta: torch.Tensor,
    step: data.FieldStep,
    mesh: Mesh,
    balance: float,
) -> torch.Tensor:
    """Computes the weak-form residual at the measured displacements.

    The residual has two parts. The first part lists the internal forces
    at the free degrees of freedom. The second part lists the reaction
    force mismatch, multiplied by sqrt(balance). The balance factor
    weights the boundary part against the bulk part.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector, shape (n_parameters,).
        step: The field step with measured displacements and reactions.
        mesh: The mesh of the step.
        balance: The weight of the reaction force part.

    Returns:
        The residual vector.
    """
    displacement = kinematics.as_tensor(step.displacement)
    forces = internal_forces(variant, theta, displacement, mesh)
    bulk = forces[mesh.free_mask]
    mismatch = reaction_forces(forces, mesh) - kinematics.as_tensor(
        step.reactions
    )
    return torch.cat([bulk, np.sqrt(balance) * mismatch])


def weak_form_jacobian(
    variant: models.ModelVariant,
    theta: torch.Tensor,
    step: data.FieldStep,
    mesh: Mesh,
    balance: float,
) -> torch.Tensor:
    """Computes the Jacobian of the weak-form residual with respect to theta.

    The derivative of the internal force f = dPi/du with respect to theta_k
    equals the gradient of dPi/dtheta_k with respect to u. So the function
    needs one backward pass per parameter, not one per residual entry.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector, shape (n_parameters,).
        step: The field step with measured displacements and reactions.
        mesh: The mesh of the step.
        balance: The weight of the reaction force part.

    Returns:
        The Jacobian, shape (n_residuals, n_parameters).
    """
    displacement = kinematics.as_tensor(step.displacement).requires_grad_(True)
    theta = theta.detach().clone().requires_grad_(True)
    energy = total_energy(variant, theta, displacement, mesh)
    (energy_gradient,) = torch.autograd.grad(energy, theta, create_graph=True)
    columns = []
    for k in range(theta.numel()):
        (sensitivity,) = torch.autograd.grad(
            energy_gradient[k], displacement, retain_graph=True
        )
        bulk = sensitivity[mesh.free_mask]
        reactions = np.sqrt(balance) * reaction_forces(sensitivity, mesh)
        columns.append(torch.cat([bulk, reactions]))
    return torch.stack(columns, dim=1)


def homogeneous_initial_guess(step: data.FieldStep) -> np.ndarray:
    """Builds a homogeneous displacement field from the boundary values.

    For each direction i, the guess is u_i = a_i X_i. The factor a_i is the
    least-squares fit to the prescribed values in direction i. The guess
    keeps every element far from inversion.

    Args:
        step: The field step with the prescribed boundary displacements.

    Returns:
        The initial nodal displacements, shape (n_nodes, 2).
    """
    guess = np.zeros_like(step.displacement)
    for direction in range(2):
        prescribed = step.dirichlet_codes[:, direction] != 0
        coordinate = step.coordinates[prescribed, direction]
        values = step.displacement[prescribed, direction]
        denominator = float(np.sum(coordinate**2))
        factor = (
            float(np.sum(coordinate * values)) / denominator
            if denominator
            else 0.0
        )
        guess[:, direction] = factor * step.coordinates[:, direction]
    prescribed_mask = step.dirichlet_codes != 0
    guess[prescribed_mask] = step.displacement[prescribed_mask]
    return guess


def solve_equilibrium(
    variant: models.ModelVariant,
    theta: np.ndarray,
    step: data.FieldStep,
    initial_displacement: np.ndarray | None = None,
    max_iterations: int = 5000,
    tolerance: float = 1e-9,
    variable_scale: float = 1e-2,
) -> EquilibriumResult:
    """Solves the forward boundary value problem for given parameters.

    The solver minimizes the total potential energy over the free degrees
    of freedom with L-BFGS-B. The prescribed degrees of freedom keep the
    measured boundary displacements.

    L-BFGS-B makes its first step along the gradient, with a step length of
    1 in the optimizer variables. A step of that size inverts small
    elements. So the optimizer variables are the free displacements divided
    by `variable_scale`. Then the first step moves the nodes by about
    `variable_scale`.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector, shape (n_parameters,).
        step: The field step. Only the boundary values and the mesh are
            used, not the measured interior displacements.
        initial_displacement: The start point, shape (n_nodes, 2). If None,
            the solver uses `homogeneous_initial_guess`.
        max_iterations: The maximum number of L-BFGS-B iterations.
        tolerance: The tolerance on the largest internal force.
        variable_scale: The scale of the optimizer variables.

    Returns:
        The equilibrium displacements and the solver statistics.
    """
    mesh = mesh_from_step(step)
    theta_tensor = kinematics.as_tensor(theta)
    if initial_displacement is None:
        initial_displacement = homogeneous_initial_guess(step)
    start = kinematics.as_tensor(initial_displacement).flatten()
    free_index = torch.nonzero(mesh.free_mask.flatten()).squeeze(1)

    def energy_and_gradient(
        scaled_free: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        free_values = kinematics.as_tensor(scaled_free * variable_scale)
        free_values.requires_grad_(True)
        displacement = start.index_put((free_index,), free_values)
        energy = total_energy(
            variant, theta_tensor, displacement.reshape(-1, 2), mesh
        )
        (gradient,) = torch.autograd.grad(energy, free_values)
        return float(energy.detach()), gradient.numpy() * variable_scale

    result = scipy.optimize.minimize(
        energy_and_gradient,
        start[free_index].numpy() / variable_scale,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "gtol": tolerance * variable_scale,
            "ftol": 1e-15,
        },
    )
    solution = start.clone()
    solution[free_index] = kinematics.as_tensor(result.x * variable_scale)
    return EquilibriumResult(
        displacement=solution.reshape(-1, 2).numpy(),
        iterations=int(result.nit),
        gradient_norm=float(np.max(np.abs(result.jac)) / variable_scale),
        success=bool(result.success),
        message=str(result.message),
    )


def predicted_reactions(
    variant: models.ModelVariant,
    theta: np.ndarray,
    displacement: np.ndarray,
    step: data.FieldStep,
) -> np.ndarray:
    """Computes the reaction forces of a displacement field.

    Args:
        variant: A compressible model variant.
        theta: The parameter vector, shape (n_parameters,).
        displacement: Nodal displacements, shape (n_nodes, 2).
        step: The field step that defines the mesh and the groups.

    Returns:
        One reaction force for each group, shape (n_groups,).
    """
    mesh = mesh_from_step(step)
    forces = internal_forces(
        variant,
        kinematics.as_tensor(theta),
        kinematics.as_tensor(displacement),
        mesh,
    )
    return reaction_forces(forces, mesh).detach().numpy()
