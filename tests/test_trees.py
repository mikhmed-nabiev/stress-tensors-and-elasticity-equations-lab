"""Tests for hyperfit.trees."""

import pytest
import sympy as sp
import torch

from hyperfit import models, trees


def test_compiled_tree_gives_sympy_expression():
    variant = models.yeoh_variant(2)
    i1b, c10, c20 = sp.symbols("I1b C10 C20")
    expected = c10 * (i1b - 3) + c20 * (i1b - 3) ** 2
    assert sp.simplify(variant.sympy_expression() - expected) == 0


def test_compiled_tree_accepts_numpy_and_torch():
    variant = models.yeoh_variant(3)
    arguments = [4.0, 0.2, -0.01, 0.001]
    numeric = variant.energy_function(*arguments)
    tensor = variant.energy_function(*[torch.tensor(a) for a in arguments])
    assert float(tensor) == pytest.approx(numeric)
    assert numeric == pytest.approx(0.2 - 0.01 + 0.001)


def test_children_indices_of_small_tree():
    primitive_set = trees.build_primitive_set(["x"])
    tree = trees.parse_tree("add(mul(x, 2), 3)", primitive_set)
    assert trees.children_indices(tree) == [[1, 4], [2, 3], [], [], []]


def test_tree_edit_distance_properties():
    yeoh1 = models.yeoh_variant(1).tree
    yeoh2 = models.yeoh_variant(2).tree
    ogden = models.ogden_variant(2).tree
    assert trees.tree_edit_distance(yeoh1, yeoh1) == 0
    assert trees.tree_edit_distance(yeoh1, yeoh2) == trees.tree_edit_distance(
        yeoh2, yeoh1
    )
    assert (
        0
        < trees.tree_edit_distance(yeoh1, yeoh2)
        < trees.tree_edit_distance(yeoh1, ogden)
    )


def test_node_label_of_constant_and_argument():
    primitive_set = trees.build_primitive_set(["x"])
    tree = trees.parse_tree("sub(x, 3)", primitive_set)
    assert [trees.node_label(node) for node in tree] == ["sub", "x", "3"]
