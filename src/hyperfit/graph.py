"""The governing equation as a directed acyclic graph (DAG).

A DAG node is a leaf or an operation. A leaf is an observed variable, a
parameter or a constant. An operation node takes the values of its input
nodes. Equal subexpressions map to one node, so a shared subexpression
appears once and has more than one consumer. A consumer of a node is an
operation that uses the node as an input.

The adjacency matrix follows the convention of the course example:
Z[i, j] = 1 if node j is an input of node i.
"""

import dataclasses

import networkx as nx
import numpy as np

from hyperfit import models, trees

OBSERVED = "observed"
PARAMETER = "parameter"
CONSTANT = "constant"
OPERATION = "operation"

_OPERATION_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "×",
    "div": "÷",
    "pow": "^",
}


@dataclasses.dataclass(frozen=True)
class DagNode:
    """One node of the DAG.

    Attributes:
        index: The position of the node in the DAG.
        kind: One of OBSERVED, PARAMETER, CONSTANT or OPERATION.
        label: The variable name, the constant or the operation symbol.
        inputs: The indices of the input nodes, in argument order.
    """

    index: int
    kind: str
    label: str
    inputs: tuple[int, ...] = ()


class ExpressionDag:
    """A DAG that merges equal subexpressions into one node."""

    def __init__(self) -> None:
        """Creates an empty DAG."""
        self.nodes: list[DagNode] = []
        self._lookup: dict[tuple[str, str, tuple[int, ...]], int] = {}

    def _add(self, kind: str, label: str, inputs: tuple[int, ...]) -> int:
        """Returns the index of a node, and creates the node if it is new.

        Args:
            kind: The node kind.
            label: The node label.
            inputs: The indices of the input nodes.

        Returns:
            The node index.
        """
        key = (kind, label, inputs)
        if key not in self._lookup:
            node = DagNode(len(self.nodes), kind, label, inputs)
            self.nodes.append(node)
            self._lookup[key] = node.index
        return self._lookup[key]

    def leaf(self, kind: str, label: str) -> int:
        """Adds a leaf node, or finds the equal leaf node.

        Args:
            kind: OBSERVED, PARAMETER or CONSTANT.
            label: The variable name or the constant.

        Returns:
            The node index.
        """
        return self._add(kind, label, ())

    def operation(self, label: str, *inputs: int) -> int:
        """Adds an operation node, or finds the equal operation node.

        Args:
            label: The operation symbol.
            *inputs: The indices of the input nodes, in argument order.

        Returns:
            The node index.
        """
        return self._add(OPERATION, label, tuple(inputs))

    def consumer_counts(self) -> np.ndarray:
        """Counts the consumers of each node.

        Returns:
            One count per node.
        """
        counts = np.zeros(len(self.nodes), dtype=int)
        for node in self.nodes:
            for input_index in set(node.inputs):
                counts[input_index] += 1
        return counts

    def shared_nodes(self, include_constants: bool = False) -> list[int]:
        """Lists the nodes with more than one consumer.

        Args:
            include_constants: If False, skip constant leaves. Constants
                such as 2 or 3 are shared in almost every expression, so
                they carry no structural information.

        Returns:
            The indices of the shared nodes.
        """
        shared = np.flatnonzero(self.consumer_counts() > 1)
        return [
            int(i)
            for i in shared
            if include_constants or self.nodes[i].kind != CONSTANT
        ]

    def adjacency_matrix(self) -> np.ndarray:
        """Builds the adjacency matrix Z.

        Returns:
            Z with Z[i, j] = 1 if node j is an input of node i.
        """
        size = len(self.nodes)
        matrix = np.zeros((size, size), dtype=int)
        for node in self.nodes:
            for input_index in node.inputs:
                matrix[node.index, input_index] = 1
        return matrix

    def to_networkx(self) -> nx.DiGraph:
        """Converts the DAG into a networkx graph.

        Returns:
            A directed graph with an edge from each input to its consumer.
        """
        graph = nx.DiGraph()
        for node in self.nodes:
            graph.add_node(node.index, kind=node.kind, label=node.label)
        for node in self.nodes:
            for input_index in node.inputs:
                graph.add_edge(input_index, node.index)
        return graph


def add_variant(
    dag: ExpressionDag,
    variant: models.ModelVariant,
    input_nodes: dict[str, int],
) -> int:
    """Adds the strain energy tree of a variant to a DAG.

    Args:
        dag: The DAG.
        variant: The model variant.
        input_nodes: Existing nodes for some kinematic inputs, by name. Any
            other kinematic input becomes an observed leaf.

    Returns:
        The index of the root node of W.
    """
    tree = variant.tree
    children = trees.children_indices(tree)

    def build(position: int) -> int:
        node = tree[position]
        if node.arity == 0:
            name = trees.node_label(node)
            if name in input_nodes:
                return input_nodes[name]
            if name in variant.inputs:
                return dag.leaf(OBSERVED, name)
            if name in variant.parameter_names:
                return dag.leaf(PARAMETER, name)
            return dag.leaf(CONSTANT, name)
        inputs = [build(child) for child in children[position]]
        return dag.operation(_OPERATION_SYMBOLS[node.name], *inputs)

    return build(0)


def build_variants_dag(
    variants: list[models.ModelVariant],
) -> tuple[ExpressionDag, dict[str, int]]:
    """Builds one DAG with the strain energy of several variants.

    Equal subexpressions of different variants map to one node. So the DAG
    shows how the variants share their terms.

    Args:
        variants: The model variants.

    Returns:
        The DAG and the root node index of each variant, by variant name.
    """
    dag = ExpressionDag()
    roots = {
        variant.name: add_variant(dag, variant, {}) for variant in variants
    }
    return dag, roots


def build_equilibrium_dag(
    variant: models.ModelVariant,
) -> tuple[ExpressionDag, dict[str, int]]:
    """Builds the DAG of the balance law Div P = 0 for one variant.

    The chain is: u -> F = I + grad u -> (I1b, J) -> W -> P = dW/dF -> Div P.

    Args:
        variant: A model variant with the inputs I1b and, optionally, J.

    Returns:
        The DAG and the indices of the named nodes "u", "F", "W", "P" and
        "Div P".
    """
    dag = ExpressionDag()
    displacement = dag.leaf(OBSERVED, "u")
    identity = dag.leaf(CONSTANT, "I")
    displacement_gradient = dag.operation("∇", displacement)
    deformation = dag.operation("+", identity, displacement_gradient)
    cauchy_green = dag.operation("FᵀF", deformation)
    first_invariant = dag.operation("tr", cauchy_green)
    volume_ratio = dag.operation("det", deformation)
    exponent = dag.leaf(CONSTANT, "-2/3")
    volume_power = dag.operation("^", volume_ratio, exponent)
    isochoric_invariant = dag.operation("×", volume_power, first_invariant)
    energy = add_variant(
        dag, variant, {"I1b": isochoric_invariant, "J": volume_ratio}
    )
    stress = dag.operation("∂/∂F", energy, deformation)
    divergence = dag.operation("Div", stress)
    named = {
        "u": displacement,
        "F": deformation,
        "W": energy,
        "P": stress,
        "Div P": divergence,
    }
    return dag, named


def _layer_depths(dag: ExpressionDag) -> dict[int, int]:
    """Computes the layer of each node.

    Leaves sit in layer 0. An operation sits one layer to the right of its
    deepest input.

    Args:
        dag: The DAG.

    Returns:
        The layer of each node.
    """
    depth: dict[int, int] = {}
    for node in dag.nodes:  # Inputs always come before their consumers.
        depth[node.index] = 1 + max((depth[i] for i in node.inputs), default=-1)
    return depth


def _spread(members: list[int]) -> dict[int, float]:
    """Spreads the nodes of one layer evenly around y = 0.

    Args:
        members: The node indices, from top to bottom.

    Returns:
        The y position of each node.
    """
    return {
        index: -(rank - (len(members) - 1) / 2.0)
        for rank, index in enumerate(members)
    }


def layer_positions(dag: ExpressionDag) -> dict[int, tuple[float, float]]:
    """Places the nodes in layers from left to right.

    The order inside each layer follows the barycenter heuristic: a node
    moves to the mean height of its neighbors. This order reduces edge
    crossings. One pass orders the operations by their inputs. A second
    pass orders the leaves by their consumers, and the operations again.

    Args:
        dag: The DAG.

    Returns:
        The (x, y) position of each node.
    """
    depth = _layer_depths(dag)
    layers: dict[int, list[int]] = {}
    for index, layer in depth.items():
        layers.setdefault(layer, []).append(index)
    consumers: dict[int, list[int]] = {node.index: [] for node in dag.nodes}
    for node in dag.nodes:
        for input_index in node.inputs:
            consumers[input_index].append(node.index)

    height = _spread(layers[0])
    for _ in range(2):
        for layer in sorted(layers)[1:]:
            members = sorted(
                layers[layer],
                key=lambda i: (
                    -np.mean([height[j] for j in dag.nodes[i].inputs])
                ),
            )
            height.update(_spread(members))
        leaves = sorted(
            layers[0],
            key=lambda i: -np.mean([height[j] for j in consumers[i]] or [0.0]),
        )
        height.update(_spread(leaves))
    for layer in sorted(layers)[1:]:
        members = sorted(
            layers[layer],
            key=lambda i: -np.mean([height[j] for j in dag.nodes[i].inputs]),
        )
        height.update(_spread(members))
    return {index: (float(depth[index]), height[index]) for index in depth}
