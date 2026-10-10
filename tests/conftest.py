"""Shared test fixtures: small synthetic files and a small mesh."""

import pathlib

import numpy as np
import openpyxl
import pandas as pd
import pytest

from hyperfit import data

# Unit square with one interior node. Corner nodes 0-3, center node 4.
SQUARE_COORDINATES = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
)
SQUARE_CONNECTIVITY = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])


def triangle_gradients(
    coordinates: np.ndarray, connectivity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Computes linear shape function gradients and element areas.

    Args:
        coordinates: Nodal coordinates, shape (n_nodes, 2).
        connectivity: Node indices, shape (n_elements, 3).

    Returns:
        The gradients, shape (n_elements, 3, 2), and the areas.
    """
    gradients = []
    areas = []
    for nodes in connectivity:
        matrix = np.column_stack([np.ones(3), coordinates[nodes]])
        inverse = np.linalg.inv(matrix)
        gradients.append(inverse[1:, :].T)
        areas.append(0.5 * abs(np.linalg.det(matrix)))
    return np.array(gradients), np.array(areas)


def affine_displacement(gradient: np.ndarray) -> np.ndarray:
    """Returns u = G X at the square nodes.

    Args:
        gradient: The displacement gradient G, shape (2, 2).

    Returns:
        Nodal displacements, shape (5, 2).
    """
    return SQUARE_COORDINATES @ gradient.T


def make_square_step(
    displacement: np.ndarray, reactions: np.ndarray
) -> data.FieldStep:
    """Builds a field step on the unit square mesh.

    The left edge has code 1 in x, the right edge code 2 in x, the bottom
    edge code 3 in y and the top edge code 4 in y. Corner nodes carry a
    code in both directions. The center node is free.

    Args:
        displacement: Nodal displacements, shape (5, 2).
        reactions: Reaction forces of the four groups.

    Returns:
        The field step.
    """
    gradients, areas = triangle_gradients(
        SQUARE_COORDINATES, SQUARE_CONNECTIVITY
    )
    codes = np.array([[1, 3], [2, 3], [2, 4], [1, 4], [0, 0]])
    zeros_elements = np.zeros((len(SQUARE_CONNECTIVITY), 2, 2))
    return data.FieldStep(
        load_step=1,
        coordinates=SQUARE_COORDINATES.copy(),
        displacement=displacement,
        dirichlet_codes=codes,
        connectivity=SQUARE_CONNECTIVITY.copy(),
        shape_gradients=gradients,
        weights=areas,
        reactions=reactions,
        reference_internal_forces=np.zeros_like(displacement),
        reference_deformation_gradient=zeros_elements,
        reference_stress=zeros_elements,
    )


@pytest.fixture
def square_step() -> data.FieldStep:
    """A biaxially stretched unit square with zero reactions."""
    gradient = np.array([[0.10, 0.0], [0.0, 0.05]])
    return make_square_step(affine_displacement(gradient), np.zeros(4))


@pytest.fixture
def srmd_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """A small file in the SRMD CSV format."""
    lines = [
        "PARAMETER;INFO;URL;;",
        "Type of test;Pull-to-failure;;;",
        "Material;TestRubber;;;",
        ";;;;",
        "Time (s);True Strain;True Stress (MPa);Engineering Strain;"
        "Engineering Stress (MPa)",
        "0.0;0.0;0.0;0.0;0.0",
        "1.0;0.0953;0.011;0.1;0.010",
        "2.0;0.1823;0.024;0.2;0.020",
        "3.0;0.2624;0.039;0.3;0.030",
        "4.0;0.3365;0.034;0.4;0.024",
    ]
    path = tmp_path / "TestRubber.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def treloar_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """A small workbook in the layout of the Treloar sheet."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["Treloar44 20"] * 9)
    sheet.append(["UT", "UT", "UT", "ET", "ET", "ET", "PS", "PS", "PS"])
    sheet.append(
        ["F11", "T11", "P11", "F11", "P11", "P11", "F11", "T11", "P11"]
    )
    sheet.append(["[-]"] * 9)
    sheet.append([None] * 9)
    sheet.append([1, 0, 0, 1, 0, 0, 1, 0, 0])
    sheet.append([1.5, 4.0, 0.39, 1.2, 3.0, 0.35, 1.3, 3.0, 0.29])
    sheet.append([2.0, 6.0, 0.59, 1.5, 4.0, 0.59, None, None, None])
    path = tmp_path / "treloar.xlsx"
    workbook.save(path)
    return path


@pytest.fixture
def euclid_directory(
    tmp_path: pathlib.Path, square_step: data.FieldStep
) -> pathlib.Path:
    """A directory with one load step in the EUCLID CSV format."""
    step_directory = tmp_path / "7"
    step_directory.mkdir()
    step = square_step
    nodes = pd.DataFrame(
        {
            "id": np.arange(5),
            "x": step.coordinates[:, 0],
            "y": step.coordinates[:, 1],
            "ux": step.displacement[:, 0],
            "uy": step.displacement[:, 1],
            "fintx": 0.0,
            "finty": 0.0,
            "bcx": step.dirichlet_codes[:, 0],
            "bcy": step.dirichlet_codes[:, 1],
        }
    )
    nodes.to_csv(step_directory / "output_nodes.csv", index=False)
    element_columns = {
        "node1": step.connectivity[:, 0],
        "node2": step.connectivity[:, 1],
        "node3": step.connectivity[:, 2],
    }
    for name in ("Fxx", "Fxy", "Fyx", "Fyy", "Pxx", "Pxy", "Pyx", "Pyy"):
        element_columns[name] = 1.0 if name in ("Fxx", "Fyy") else 0.0
    pd.DataFrame(element_columns).to_csv(
        step_directory / "output_elements.csv", index=False
    )
    integrator = {}
    for a in range(3):
        for axis, column in (("x", 0), ("y", 1)):
            integrator[f"gradNa_node{a + 1}_{axis}"] = step.shape_gradients[
                :, a, column
            ]
    integrator["qpWeight"] = step.weights
    pd.DataFrame(integrator).to_csv(
        step_directory / "output_integrator.csv", index=False
    )
    pd.DataFrame({"forces": [-1.0, 1.0, -2.0, 2.0]}).to_csv(
        step_directory / "output_reactions.csv", index=False
    )
    return tmp_path
