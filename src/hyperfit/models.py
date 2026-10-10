"""Strain energy model variants.

A model variant is one strain energy function W. The variants in this
project are:

* Yeoh of order N: W = sum_{i=1..N} C_i0 (I1b - 3)^i.
  Order 1 is the neo-Hookean model.
* Compressible Yeoh of order N: the Yeoh terms plus D1 (J - 1)^2.
* Ogden with M terms:
  W = sum_{p=1..M} (mu_p / alpha_p) (l1^alpha_p + l2^alpha_p + l3^alpha_p - 3).

Each variant stores its formula as a DEAP prefix expression. The tree is
the single source of the formula. The SymPy form, the torch function and
the graph all come from the same tree.
"""

import dataclasses
import functools
from collections.abc import Callable

import sympy as sp
import torch
from deap import gp

from hyperfit import trees

YEOH = "yeoh"
OGDEN = "ogden"


@dataclasses.dataclass(frozen=True)
class ModelVariant:
    """One strain energy function and its parameters.

    Attributes:
        name: A short identifier, for example "yeoh3".
        title: A human-readable name for tables and plots.
        family: The model family. One of `YEOH` or `OGDEN`.
        expression: The DEAP prefix expression of W.
        inputs: The kinematic input names, in call order.
        parameter_names: The parameter names, in call order. This order
            defines the order of the parameter vector.
        compressible: True if W has a volumetric term in J.
    """

    name: str
    title: str
    family: str
    expression: str
    inputs: tuple[str, ...]
    parameter_names: tuple[str, ...]
    compressible: bool

    @property
    def argument_names(self) -> tuple[str, ...]:
        """The tree arguments: kinematic inputs, then parameters."""
        return self.inputs + self.parameter_names

    @property
    def n_parameters(self) -> int:
        """The length of the parameter vector."""
        return len(self.parameter_names)

    @functools.cached_property
    def primitive_set(self) -> gp.PrimitiveSet:
        """The DEAP primitive set of the variant."""
        return trees.build_primitive_set(self.argument_names)

    @functools.cached_property
    def tree(self) -> gp.PrimitiveTree:
        """The DEAP tree of W."""
        return trees.parse_tree(self.expression, self.primitive_set)

    @functools.cached_property
    def energy_function(self) -> Callable:
        """W as a Python function of the tree arguments."""
        return trees.compile_tree(self.tree, self.primitive_set)

    def sympy_expression(self) -> sp.Expr:
        """Returns W as a SymPy expression."""
        return trees.tree_to_sympy(
            self.tree, self.primitive_set, self.argument_names
        )


def _yeoh_term(order: int) -> str:
    """Builds the Yeoh term of one order as a prefix expression.

    Args:
        order: The order i of the term C_i0 (I1b - 3)^i.

    Returns:
        The prefix expression of the term.
    """
    deviation = "sub(I1b, 3)"
    if order == 1:
        return f"mul(C10, {deviation})"
    return f"mul(C{order}0, pow({deviation}, {order}))"


def yeoh_expression(order: int, compressible: bool) -> str:
    """Builds the prefix expression of a Yeoh strain energy.

    Each order adds one term to the expression of the previous order. So
    the tree of order N contains the tree of order N - 1 as a subtree.

    Args:
        order: The highest order N, from 1 to 3.
        compressible: If True, add the volumetric term D1 (J - 1)^2.

    Returns:
        The prefix expression of W.
    """
    expression = _yeoh_term(1)
    for term_order in range(2, order + 1):
        expression = f"add({expression}, {_yeoh_term(term_order)})"
    if compressible:
        expression = f"add({expression}, mul(D1, pow(sub(J, 1), 2)))"
    return expression


def yeoh_variant(order: int, compressible: bool = False) -> ModelVariant:
    """Creates a Yeoh model variant.

    Args:
        order: The highest order N, from 1 to 3.
        compressible: If True, add the volumetric term D1 (J - 1)^2.

    Returns:
        The model variant.

    Raises:
        ValueError: The order is not 1, 2 or 3.
    """
    if order not in (1, 2, 3):
        raise ValueError(f"The Yeoh order must be 1, 2 or 3, not {order}.")
    parameter_names = tuple(f"C{i}0" for i in range(1, order + 1))
    inputs = ("I1b",)
    name = f"yeoh{order}"
    title = f"Yeoh N={order}" + (" (neo-Hookean)" if order == 1 else "")
    if compressible:
        parameter_names += ("D1",)
        inputs += ("J",)
        name += "_compressible"
        title += " + volumetric"
    return ModelVariant(
        name=name,
        title=title,
        family=YEOH,
        expression=yeoh_expression(order, compressible),
        inputs=inputs,
        parameter_names=parameter_names,
        compressible=compressible,
    )


def _ogden_term(term: int) -> str:
    """Builds one Ogden term as a prefix expression.

    Args:
        term: The term number p, starting at 1.

    Returns:
        The prefix expression of (mu_p / alpha_p) (sum of l_k^alpha_p - 3).
    """
    alpha = f"alpha{term}"
    stretch_sum = (
        f"add(add(pow(l1, {alpha}), pow(l2, {alpha})), pow(l3, {alpha}))"
    )
    return f"mul(div(mu{term}, {alpha}), sub({stretch_sum}, 3))"


def ogden_variant(n_terms: int) -> ModelVariant:
    """Creates an incompressible Ogden model variant.

    Args:
        n_terms: The number of terms M.

    Returns:
        The model variant. The parameter order is mu1, alpha1, mu2, ...
    """
    expression = _ogden_term(1)
    for term in range(2, n_terms + 1):
        expression = f"add({expression}, {_ogden_term(term)})"
    parameter_names: tuple[str, ...] = ()
    for term in range(1, n_terms + 1):
        parameter_names += (f"mu{term}", f"alpha{term}")
    return ModelVariant(
        name=f"ogden{n_terms}",
        title=f"Ogden M={n_terms}",
        family=OGDEN,
        expression=expression,
        inputs=("l1", "l2", "l3"),
        parameter_names=parameter_names,
        compressible=False,
    )


def homogeneous_variants() -> list[ModelVariant]:
    """Returns the incompressible variants for homogeneous tests."""
    return [yeoh_variant(1), yeoh_variant(2), yeoh_variant(3), ogden_variant(2)]


def field_variants() -> list[ModelVariant]:
    """Returns the compressible variants for the field test."""
    return [yeoh_variant(order, compressible=True) for order in (1, 2, 3)]


def strain_energy(
    variant: ModelVariant,
    kinematic_inputs: dict[str, torch.Tensor],
    theta: torch.Tensor,
) -> torch.Tensor:
    """Evaluates W at many material points.

    Args:
        variant: The model variant.
        kinematic_inputs: Tensors for every name in `variant.inputs`.
        theta: The parameter vector, shape (n_parameters,).

    Returns:
        W at each point, with the shape of the kinematic inputs.
    """
    arguments = [kinematic_inputs[name] for name in variant.inputs]
    arguments += [theta[i] for i in range(variant.n_parameters)]
    return variant.energy_function(*arguments)
