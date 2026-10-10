"""Expression trees for strain energy functions.

The module follows the supervisor's tool recommendation:

* DEAP stores each strain energy function as a tree. A tree is a list of
  nodes in prefix order. A primitive node is an operation. A terminal
  node is a variable or a constant.
* The same compiled tree accepts SymPy symbols, NumPy arrays or torch
  tensors, because every primitive is a Python operator.
* zss computes the tree edit distance. The tree edit distance is the
  smallest number of node insertions, deletions and relabels that turns
  one tree into another tree.
"""

import operator
from collections.abc import Callable, Sequence

import sympy as sp
import zss
from deap import gp

PRIMITIVES = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "pow": operator.pow,
}
CONSTANTS = (1, 2, 3)


def build_primitive_set(argument_names: Sequence[str]) -> gp.PrimitiveSet:
    """Builds the DEAP primitive set for a strain energy function.

    Args:
        argument_names: The names of the tree arguments, in call order.
            The arguments are the kinematic inputs and then the parameters.

    Returns:
        A primitive set with the arithmetic operations, the arguments and
        the integer constants 1, 2 and 3.
    """
    primitive_set = gp.PrimitiveSet("W", len(argument_names))
    primitive_set.renameArguments(
        **{f"ARG{i}": name for i, name in enumerate(argument_names)}
    )
    for name, function in PRIMITIVES.items():
        primitive_set.addPrimitive(function, 2, name=name)
    for constant in CONSTANTS:
        primitive_set.addTerminal(constant, name=str(constant))
    return primitive_set


def parse_tree(
    expression: str, primitive_set: gp.PrimitiveSet
) -> gp.PrimitiveTree:
    """Parses a prefix expression into a DEAP tree.

    Args:
        expression: The expression, for example "mul(C10, sub(I1b, 3))".
        primitive_set: The primitive set that defines every name.

    Returns:
        The DEAP tree.
    """
    return gp.PrimitiveTree.from_string(expression, primitive_set)


def compile_tree(
    tree: gp.PrimitiveTree, primitive_set: gp.PrimitiveSet
) -> Callable:
    """Compiles a DEAP tree into a Python function.

    Args:
        tree: The DEAP tree.
        primitive_set: The primitive set of the tree.

    Returns:
        A function. It takes the arguments in the order of the primitive
        set and returns the value of the expression.
    """
    return gp.compile(tree, primitive_set)


def tree_to_sympy(
    tree: gp.PrimitiveTree,
    primitive_set: gp.PrimitiveSet,
    argument_names: Sequence[str],
) -> sp.Expr:
    """Converts a DEAP tree into a SymPy expression.

    Args:
        tree: The DEAP tree.
        primitive_set: The primitive set of the tree.
        argument_names: The argument names, in call order.

    Returns:
        The SymPy expression.
    """
    function = compile_tree(tree, primitive_set)
    symbols = sp.symbols(list(argument_names))
    return sp.sympify(function(*symbols))


def node_label(node: gp.Primitive | gp.Terminal) -> str:
    """Returns a short text label for one DEAP node.

    Args:
        node: A primitive or a terminal.

    Returns:
        The primitive name, the argument name or the constant value.
    """
    if isinstance(node, gp.Primitive):
        return node.name
    return str(node.value)


def children_indices(tree: gp.PrimitiveTree) -> list[list[int]]:
    """Finds the children of every node in a prefix-order tree.

    Args:
        tree: The DEAP tree.

    Returns:
        A list with one entry per node. Each entry lists the positions of
        the node children, from left to right.
    """
    children: list[list[int]] = [[] for _ in tree]
    open_nodes: list[int] = []
    for position, node in enumerate(tree):
        if open_nodes:
            parent = open_nodes[-1]
            children[parent].append(position)
            if len(children[parent]) == tree[parent].arity:
                open_nodes.pop()
        if node.arity > 0:
            open_nodes.append(position)
    return children


def tree_to_zss(tree: gp.PrimitiveTree) -> zss.Node:
    """Converts a DEAP tree into a zss tree.

    Args:
        tree: The DEAP tree.

    Returns:
        The root node of the zss tree.
    """
    children = children_indices(tree)
    zss_nodes = [zss.Node(node_label(node)) for node in tree]
    for position, child_positions in enumerate(children):
        for child in child_positions:
            zss_nodes[position].addkid(zss_nodes[child])
    return zss_nodes[0]


def tree_edit_distance(
    first: gp.PrimitiveTree, second: gp.PrimitiveTree
) -> int:
    """Computes the Zhang-Shasha tree edit distance between two trees.

    Every insertion, deletion and relabel costs 1.

    Args:
        first: The first DEAP tree.
        second: The second DEAP tree.

    Returns:
        The tree edit distance.
    """
    return int(zss.simple_distance(tree_to_zss(first), tree_to_zss(second)))
