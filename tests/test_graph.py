"""Tests for hyperfit.graph."""

import networkx as nx

from hyperfit import graph, models


def test_equal_nodes_merge():
    dag = graph.ExpressionDag()
    first = dag.leaf(graph.OBSERVED, "x")
    assert dag.leaf(graph.OBSERVED, "x") == first
    operation = dag.operation("+", first, first)
    assert dag.operation("+", first, first) == operation
    assert len(dag.nodes) == 2


def test_yeoh_variants_share_subtrees():
    variants = [models.yeoh_variant(order) for order in (1, 2, 3)]
    dag, roots = graph.build_variants_dag(variants)
    labels = {node.index: node for node in dag.nodes}
    yeoh3_root = labels[roots["yeoh3"]]
    assert roots["yeoh2"] in yeoh3_root.inputs
    assert roots["yeoh1"] in labels[roots["yeoh2"]].inputs
    deviation = [
        n for n in dag.nodes if n.label == "-" and n.kind == graph.OPERATION
    ]
    assert len(deviation) == 1
    assert deviation[0].index in dag.shared_nodes()


def test_adjacency_matrix_matches_edges():
    dag, _ = graph.build_equilibrium_dag(
        models.yeoh_variant(2, compressible=True)
    )
    matrix = dag.adjacency_matrix()
    network = dag.to_networkx()
    assert matrix.sum() == network.number_of_edges()
    assert nx.is_directed_acyclic_graph(network)
    for node in dag.nodes:
        for input_index in node.inputs:
            assert matrix[node.index, input_index] == 1


def test_equilibrium_dag_structure():
    dag, named = graph.build_equilibrium_dag(
        models.yeoh_variant(1, compressible=True)
    )
    assert named["F"] in dag.shared_nodes()
    assert dag.nodes[named["Div P"]].inputs == (named["P"],)
    assert dag.nodes[named["P"]].inputs == (named["W"], named["F"])


def test_layer_positions_place_inputs_left():
    dag, _ = graph.build_equilibrium_dag(
        models.yeoh_variant(3, compressible=True)
    )
    positions = graph.layer_positions(dag)
    for node in dag.nodes:
        for input_index in node.inputs:
            assert positions[input_index][0] < positions[node.index][0]


def test_shared_nodes_skip_constants_by_default():
    dag, _ = graph.build_variants_dag([models.yeoh_variant(3)])
    constants = [n.index for n in dag.nodes if n.kind == graph.CONSTANT]
    assert not set(constants) & set(dag.shared_nodes())
    assert set(constants) & set(dag.shared_nodes(include_constants=True))
