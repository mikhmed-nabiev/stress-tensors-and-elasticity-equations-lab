"""Parameter identification with three optimizers.

Every fit minimizes the cost 0.5 * ||r(theta)||^2 of a residual function r.
A residual function maps a parameter vector to a residual vector, both as
torch tensors.

The three optimizers are:

* Bounded linear least squares. It finds the exact global minimum when r
  is linear in theta. This is true for every Yeoh variant.
* Levenberg-Marquardt (LM). LM is a gradient-based least-squares method.
  torch autograd gives its Jacobian. LM ignores the bounds.
* CMA-ES (covariance matrix adaptation evolution strategy). CMA-ES is a
  derivative-free global optimizer. It samples candidate vectors from a
  normal distribution and adapts the distribution to the best samples.
"""

import dataclasses
from collections.abc import Callable

import cma
import numpy as np
import scipy.optimize
import torch

from hyperfit import data, fem, kinematics, mechanics, models

ResidualFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class FitResult:
    """The result of one fit.

    Attributes:
        method: The optimizer name.
        theta: The fitted parameter vector.
        cost: The final cost 0.5 * ||r||^2.
        n_evaluations: The number of residual evaluations.
        success: True if the optimizer reports convergence.
        message: The optimizer message.
        history: The best cost after each CMA-ES generation. Empty for the
            other optimizers.
    """

    method: str
    theta: np.ndarray
    cost: float
    n_evaluations: int
    success: bool
    message: str
    history: tuple[float, ...] = ()


def homogeneous_residual_function(
    variant: models.ModelVariant, tests: list[data.HomogeneousTest]
) -> ResidualFunction:
    """Builds the relative stress residual for homogeneous tests.

    The residual of one point is (P_model - P_data) / P_data. A relative
    residual gives the small-stress points the same weight as the
    large-stress points.

    Args:
        variant: An incompressible model variant.
        tests: The homogeneous tests to fit together.

    Returns:
        The residual function.
    """
    stretches = [kinematics.as_tensor(test.stretch) for test in tests]
    stresses = [kinematics.as_tensor(test.nominal_stress) for test in tests]
    modes = [test.mode for test in tests]

    def residual(theta: torch.Tensor) -> torch.Tensor:
        parts = []
        for mode, stretch, stress in zip(
            modes, stretches, stresses, strict=True
        ):
            model = mechanics.nominal_stress(variant, mode, stretch, theta)
            parts.append((model - stress) / stress)
        return torch.cat(parts)

    return residual


def field_residual_function(
    variant: models.ModelVariant, steps: list[data.FieldStep], balance: float
) -> ResidualFunction:
    """Builds the weak-form residual for one or more field steps.

    Args:
        variant: A compressible model variant.
        steps: The field steps to fit together.
        balance: The weight of the reaction force part.

    Returns:
        The residual function.
    """
    meshes = [fem.mesh_from_step(step) for step in steps]

    def residual(theta: torch.Tensor) -> torch.Tensor:
        parts = [
            fem.weak_form_residual(variant, theta, step, mesh, balance)
            for step, mesh in zip(steps, meshes, strict=True)
        ]
        return torch.cat(parts)

    return residual


def field_jacobian_function(
    variant: models.ModelVariant, steps: list[data.FieldStep], balance: float
) -> ResidualFunction:
    """Builds the Jacobian function of the weak-form residual.

    Args:
        variant: A compressible model variant.
        steps: The field steps, in the order of `field_residual_function`.
        balance: The weight of the reaction force part.

    Returns:
        A function that maps theta to the Jacobian dr/dtheta.
    """
    meshes = [fem.mesh_from_step(step) for step in steps]

    def jacobian(theta: torch.Tensor) -> torch.Tensor:
        parts = [
            fem.weak_form_jacobian(variant, theta, step, mesh, balance)
            for step, mesh in zip(steps, meshes, strict=True)
        ]
        return torch.cat(parts)

    return jacobian


def evaluate_residual(
    residual: ResidualFunction, theta: np.ndarray
) -> np.ndarray:
    """Evaluates a residual function for a NumPy parameter vector.

    Args:
        residual: The residual function.
        theta: The parameter vector.

    Returns:
        The residual vector as a NumPy array.
    """
    return residual(kinematics.as_tensor(theta)).detach().numpy()


def evaluate_jacobian(
    residual: ResidualFunction, theta: np.ndarray
) -> np.ndarray:
    """Computes the Jacobian dr/dtheta with torch autograd.

    The function uses vectorized reverse mode. It suits residuals with a
    few thousand entries or less.

    Args:
        residual: The residual function.
        theta: The parameter vector, shape (n_parameters,).

    Returns:
        The Jacobian, shape (n_residuals, n_parameters).
    """
    jacobian = torch.autograd.functional.jacobian(
        residual, kinematics.as_tensor(theta), vectorize=True
    )
    return jacobian.detach().numpy()


def cost(residual: ResidualFunction, theta: np.ndarray) -> float:
    """Computes the cost 0.5 * ||r(theta)||^2.

    Args:
        residual: The residual function.
        theta: The parameter vector.

    Returns:
        The cost.
    """
    values = evaluate_residual(residual, theta)
    return 0.5 * float(values @ values)


def linearize(
    residual: ResidualFunction, n_parameters: int
) -> tuple[np.ndarray, np.ndarray]:
    """Writes a linear residual as r(theta) = A theta - b.

    The function evaluates r at theta = 0 and at each unit vector.

    Args:
        residual: A residual function that is linear in theta.
        n_parameters: The length of the parameter vector.

    Returns:
        The matrix A, shape (n_residuals, n_parameters), and the vector b,
        shape (n_residuals,).
    """
    offset = evaluate_residual(residual, np.zeros(n_parameters))
    columns = [
        evaluate_residual(residual, unit) - offset
        for unit in np.eye(n_parameters)
    ]
    return np.stack(columns, axis=1), -offset


def is_linear(
    residual: ResidualFunction,
    matrix: np.ndarray,
    vector: np.ndarray,
    theta: np.ndarray,
    tolerance: float = 1e-8,
) -> bool:
    """Checks the linear form A theta - b at one test vector.

    Args:
        residual: The residual function.
        matrix: The matrix A from `linearize`.
        vector: The vector b from `linearize`.
        theta: A test parameter vector.
        tolerance: The allowed relative difference.

    Returns:
        True if r(theta) equals A theta - b within the tolerance.
    """
    direct = evaluate_residual(residual, theta)
    linear = matrix @ theta - vector
    scale = max(float(np.linalg.norm(direct)), 1.0)
    return float(np.linalg.norm(direct - linear)) <= tolerance * scale


def fit_linear_least_squares(
    residual: ResidualFunction, lower: np.ndarray, upper: np.ndarray
) -> FitResult:
    """Finds the global minimum of a linear residual within bounds.

    Args:
        residual: A residual function that is linear in theta.
        lower: The lower bounds.
        upper: The upper bounds.

    Returns:
        The fit result.

    Raises:
        ValueError: The residual is not linear in theta.
    """
    n_parameters = lower.size
    matrix, vector = linearize(residual, n_parameters)
    probe = 0.1 * np.arange(1, n_parameters + 1)
    if not is_linear(residual, matrix, vector, probe):
        raise ValueError("The residual is not linear in theta.")
    result = scipy.optimize.lsq_linear(matrix, vector, bounds=(lower, upper))
    return FitResult(
        method="Linear least squares",
        theta=result.x,
        cost=cost(residual, result.x),
        n_evaluations=n_parameters + 2,
        success=bool(result.success),
        message=str(result.message),
    )


def fit_levenberg_marquardt(
    residual: ResidualFunction,
    initial: np.ndarray,
    jacobian: ResidualFunction | None = None,
    max_evaluations: int = 2000,
) -> FitResult:
    """Fits the parameters with the Levenberg-Marquardt method.

    Args:
        residual: The residual function.
        initial: The start point.
        jacobian: A function that maps theta to dr/dtheta. If None, the
            function uses `evaluate_jacobian`.
        max_evaluations: The maximum number of residual evaluations.

    Returns:
        The fit result.
    """

    def jacobian_values(theta: np.ndarray) -> np.ndarray:
        if jacobian is None:
            return evaluate_jacobian(residual, theta)
        return jacobian(kinematics.as_tensor(theta)).detach().numpy()

    result = scipy.optimize.least_squares(
        lambda theta: evaluate_residual(residual, theta),
        initial,
        jac=jacobian_values,
        method="lm",
        x_scale="jac",
        max_nfev=max_evaluations,
    )
    return FitResult(
        method="Levenberg-Marquardt",
        theta=result.x,
        cost=float(result.cost),
        n_evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def fit_cma_es(
    residual: ResidualFunction,
    initial: np.ndarray,
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sigma0: float = 0.5,
    seed: int = 1,
    max_evaluations: int = 6000,
) -> FitResult:
    """Fits the parameters with CMA-ES.

    CMA-ES works on the scaled vector z = theta / scale. Then one unit of
    z has the same meaning for every parameter.

    Args:
        residual: The residual function.
        initial: The start point.
        scale: The typical magnitude of each parameter.
        lower: The lower bounds.
        upper: The upper bounds.
        sigma0: The initial step size in units of z.
        seed: The random seed. A fixed seed makes the fit repeatable.
        max_evaluations: The maximum number of cost evaluations.

    Returns:
        The fit result with the cost history.
    """
    options = {
        "bounds": [list(lower / scale), list(upper / scale)],
        "seed": seed,
        "maxfevals": max_evaluations,
        "tolfun": 1e-14,
        "tolx": 1e-12,
        "verbose": -9,
    }
    strategy = cma.CMAEvolutionStrategy(initial / scale, sigma0, options)
    history = []
    while not strategy.stop():
        candidates = strategy.ask()
        costs = [cost(residual, candidate * scale) for candidate in candidates]
        strategy.tell(candidates, costs)
        history.append(float(strategy.result.fbest))
    theta = np.asarray(strategy.result.xbest) * scale
    return FitResult(
        method="CMA-ES",
        theta=theta,
        cost=cost(residual, theta),
        n_evaluations=int(strategy.result.evaluations),
        success=True,
        message=str(dict(strategy.stop())),
        history=tuple(history),
    )
