"""Download, cache and parse the three data sources.

The module has three parts:

* HTTP download with a local cache and SHA-256 checks.
* Parsers that convert each file format into simple data classes.
* Small preprocessing steps for homogeneous test curves.

A homogeneous test applies the same deformation to every point of the
specimen. For such a test, one stretch and one stress describe the full
state. A field test has a different deformation at each point. For a field
test, the data holds nodal displacements on a finite element mesh.
"""

import dataclasses
import hashlib
import pathlib
import shutil
import tempfile
import urllib.request

import numpy as np
import pandas as pd

from hyperfit import sources

UNIAXIAL = "uniaxial"
EQUIBIAXIAL = "equibiaxial"
PURE_SHEAR = "pure_shear"
TEST_MODES = (UNIAXIAL, EQUIBIAXIAL, PURE_SHEAR)

_DOWNLOAD_TIMEOUT_S = 60
_HASH_BLOCK_BYTES = 1 << 16


@dataclasses.dataclass(frozen=True)
class HomogeneousTest:
    """One stress-stretch curve from a homogeneous test.

    Attributes:
        source_key: The key of the data source.
        material: The material name.
        mode: The test mode. One of `TEST_MODES`.
        stretch: The stretch in the loading direction. The stretch is the
            deformed length divided by the original length.
        nominal_stress: The nominal stress in the loading direction, in the
            stress unit of the source. The nominal stress is the force
            divided by the undeformed cross-section area.
        time: The test time in seconds, or None if the source has no time.
    """

    source_key: str
    material: str
    mode: str
    stretch: np.ndarray
    nominal_stress: np.ndarray
    time: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class FieldStep:
    """Finite element data for one load step of a field test.

    The mesh uses linear triangles with one quadrature point each.

    Attributes:
        load_step: The load step number.
        coordinates: Nodal reference coordinates, shape (n_nodes, 2).
        displacement: Nodal displacements, shape (n_nodes, 2).
        dirichlet_codes: Boundary codes, shape (n_nodes, 2). Code 0 marks a
            free degree of freedom. Code k > 0 marks a prescribed degree of
            freedom that belongs to reaction force group k.
        connectivity: Node indices of each element, shape (n_elements, 3).
        shape_gradients: Gradients of the three shape functions with
            respect to the reference coordinates, shape (n_elements, 3, 2).
        weights: Quadrature weight (element area) of each element,
            shape (n_elements,).
        reactions: Measured reaction force of each group, shape (n_groups,).
            Entry k - 1 belongs to boundary code k.
        reference_internal_forces: Nodal internal forces from the source
            solver, shape (n_nodes, 2).
        reference_deformation_gradient: In-plane deformation gradient from
            the source solver, shape (n_elements, 2, 2).
        reference_stress: In-plane first Piola-Kirchhoff stress from the
            source solver, shape (n_elements, 2, 2).
    """

    load_step: int
    coordinates: np.ndarray
    displacement: np.ndarray
    dirichlet_codes: np.ndarray
    connectivity: np.ndarray
    shape_gradients: np.ndarray
    weights: np.ndarray
    reactions: np.ndarray
    reference_internal_forces: np.ndarray
    reference_deformation_gradient: np.ndarray
    reference_stress: np.ndarray


def compute_sha256(path: pathlib.Path) -> str:
    """Computes the SHA-256 checksum of a file.

    Args:
        path: The file to read.

    Returns:
        The checksum as a lowercase hexadecimal string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    remote: sources.RemoteFile, directory: pathlib.Path
) -> pathlib.Path:
    """Downloads one remote file into a directory, with a cache.

    The function skips the download if a file with the correct checksum
    already exists.

    Args:
        remote: The remote file to download.
        directory: The cache directory of the data source.

    Returns:
        The path of the local file.

    Raises:
        ValueError: The downloaded file has a wrong checksum.
    """
    target = directory / remote.local_name
    if target.exists() and compute_sha256(target) == remote.sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        temporary_path = pathlib.Path(tmp.name)
        with urllib.request.urlopen(
            remote.url, timeout=_DOWNLOAD_TIMEOUT_S
        ) as response:
            shutil.copyfileobj(response, tmp)
    checksum = compute_sha256(temporary_path)
    if checksum != remote.sha256:
        temporary_path.unlink()
        raise ValueError(
            f"Checksum mismatch for {remote.url}: "
            f"expected {remote.sha256}, received {checksum}."
        )
    temporary_path.replace(target)
    return target


def fetch_source(
    source: sources.DataSource, cache_root: pathlib.Path
) -> pathlib.Path:
    """Downloads all files of a data source into the cache.

    Args:
        source: The data source.
        cache_root: The root directory of the cache.

    Returns:
        The cache directory of the data source.
    """
    directory = cache_root / source.key
    for remote in source.files:
        download_file(remote, directory)
    return directory


def read_srmd_metadata(path: pathlib.Path) -> dict[str, str]:
    """Reads the metadata block at the top of an SRMD CSV file.

    Args:
        path: The SRMD CSV file.

    Returns:
        A mapping from metadata name to metadata value.
    """
    metadata = {}
    with open(path, encoding="utf-8") as stream:
        next(stream)  # The first line is the block header.
        for line in stream:
            fields = line.rstrip("\n").split(";")
            if not fields[0]:
                break
            metadata[fields[0]] = fields[1]
    return metadata


def read_srmd_table(path: pathlib.Path) -> pd.DataFrame:
    """Reads the measurement table of an SRMD CSV file.

    Args:
        path: The SRMD CSV file.

    Returns:
        A table with one row per measurement point.

    Raises:
        ValueError: The file has no measurement table.
    """
    with open(path, encoding="utf-8") as stream:
        lines = stream.readlines()
    header_rows = [
        i for i, line in enumerate(lines) if line.startswith("Time (s)")
    ]
    if not header_rows:
        raise ValueError(f"No measurement table in {path}.")
    return pd.read_csv(path, sep=";", skiprows=header_rows[0], encoding="utf-8")


def srmd_uniaxial_test(path: pathlib.Path, source_key: str) -> HomogeneousTest:
    """Converts an SRMD CSV file into a uniaxial homogeneous test.

    The stretch is 1 plus the engineering strain. The nominal stress is the
    engineering stress, in MPa.

    Args:
        path: The SRMD CSV file.
        source_key: The key of the data source.

    Returns:
        The uniaxial homogeneous test.
    """
    metadata = read_srmd_metadata(path)
    table = read_srmd_table(path)
    return HomogeneousTest(
        source_key=source_key,
        material=metadata.get("Material", path.stem),
        mode=UNIAXIAL,
        stretch=1.0 + table["Engineering Strain"].to_numpy(dtype=float),
        nominal_stress=table["Engineering Stress (MPa)"].to_numpy(dtype=float),
        time=table["Time (s)"].to_numpy(dtype=float),
    )


def _select_points(test: HomogeneousTest, mask: np.ndarray) -> HomogeneousTest:
    """Keeps the points of a test where the mask is True.

    Args:
        test: The homogeneous test.
        mask: A boolean array with one entry per point.

    Returns:
        A new test with the selected points.
    """
    time = None if test.time is None else test.time[mask]
    return dataclasses.replace(
        test,
        stretch=test.stretch[mask],
        nominal_stress=test.nominal_stress[mask],
        time=time,
    )


def truncate_at_peak_stress(test: HomogeneousTest) -> HomogeneousTest:
    """Removes the points after the peak stress.

    The points after the peak belong to specimen failure, not to the
    elastic response.

    Args:
        test: The homogeneous test.

    Returns:
        A new test that ends at the peak stress.
    """
    peak_index = int(np.argmax(test.nominal_stress))
    mask = np.arange(test.stretch.size) <= peak_index
    return _select_points(test, mask)


def drop_unloaded_points(test: HomogeneousTest) -> HomogeneousTest:
    """Removes the points with no stretch or no stress.

    A relative error is not defined at zero stress.

    Args:
        test: The homogeneous test.

    Returns:
        A new test with stretch > 1 and stress > 0 at every point.
    """
    mask = (test.stretch > 1.0) & (test.nominal_stress > 0.0)
    return _select_points(test, mask)


def rebuild_uniform_time(test: HomogeneousTest) -> HomogeneousTest:
    """Rebuilds a time column that has a constant sampling interval.

    The SRMD time column stops increasing after 40.9 s: it continues with
    40.1, 40.11, 40.12 and so on. The stretch increment per row stays
    constant over the full test. At a constant crosshead speed, a constant
    stretch increment means a constant sampling interval. So the function
    rebuilds the time from the row index and the first sampling interval.

    Args:
        test: A homogeneous test with time data.

    Returns:
        A new test with a strictly increasing, uniform time column.

    Raises:
        ValueError: The test has no time data.
    """
    if test.time is None:
        raise ValueError("The test has no time data.")
    interval = float(test.time[1] - test.time[0])
    time = test.time[0] + interval * np.arange(test.time.size)
    return dataclasses.replace(test, time=time)


def _check_treloar_layout(raw: pd.DataFrame) -> None:
    """Checks the column layout of the Treloar sheet.

    Args:
        raw: The first rows of the sheet, without a header.

    Raises:
        ValueError: The sheet has an unexpected layout.
    """
    expected_modes = {0: "UT", 3: "ET", 6: "PS"}
    for column, mode in expected_modes.items():
        if raw.iloc[1, column] != mode or raw.iloc[2, column] != "F11":
            raise ValueError("The Treloar sheet has an unexpected layout.")


def read_treloar_tests(
    path: pathlib.Path, source_key: str
) -> list[HomogeneousTest]:
    """Reads the three Treloar (1944) tests at 20 degrees Celsius.

    The sheet stores the stretch and the nominal stress in MPa for each
    test mode in columns A to I.

    Args:
        path: The Excel workbook.
        source_key: The key of the data source.

    Returns:
        Three homogeneous tests: uniaxial, equibiaxial and pure shear.
    """
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None, usecols="A:I")
    _check_treloar_layout(raw)
    values = raw.iloc[5:].apply(pd.to_numeric, errors="coerce")
    columns = {UNIAXIAL: (0, 2), EQUIBIAXIAL: (3, 5), PURE_SHEAR: (6, 8)}
    tests = []
    for mode, (stretch_column, stress_column) in columns.items():
        pair = values.iloc[:, [stretch_column, stress_column]].dropna()
        tests.append(
            HomogeneousTest(
                source_key=source_key,
                material="Natural rubber (Treloar 1944, 20 C)",
                mode=mode,
                stretch=pair.iloc[:, 0].to_numpy(dtype=float),
                nominal_stress=pair.iloc[:, 1].to_numpy(dtype=float),
            )
        )
    return tests


def _as_matrices(table: pd.DataFrame, prefix: str) -> np.ndarray:
    """Collects four columns xx, xy, yx, yy into 2 x 2 matrices.

    Args:
        table: The element table.
        prefix: The column prefix, for example "F" or "P".

    Returns:
        An array of shape (n_rows, 2, 2).
    """
    names = [f"{prefix}xx", f"{prefix}xy", f"{prefix}yx", f"{prefix}yy"]
    return table[names].to_numpy(dtype=float).reshape(-1, 2, 2)


def read_field_step(directory: pathlib.Path, load_step: int) -> FieldStep:
    """Reads the EUCLID finite element data of one load step.

    Args:
        directory: The cache directory of the EUCLID data source.
        load_step: The load step number.

    Returns:
        The finite element data of the load step.
    """
    step_directory = directory / str(load_step)
    nodes = pd.read_csv(step_directory / "output_nodes.csv")
    elements = pd.read_csv(step_directory / "output_elements.csv")
    integrator = pd.read_csv(step_directory / "output_integrator.csv")
    reactions = pd.read_csv(step_directory / "output_reactions.csv")

    gradient_columns = [
        f"gradNa_node{a}_{axis}" for a in (1, 2, 3) for axis in ("x", "y")
    ]
    return FieldStep(
        load_step=load_step,
        coordinates=nodes[["x", "y"]].to_numpy(dtype=float),
        displacement=nodes[["ux", "uy"]].to_numpy(dtype=float),
        dirichlet_codes=nodes[["bcx", "bcy"]].round().to_numpy(dtype=int),
        connectivity=elements[["node1", "node2", "node3"]]
        .round()
        .to_numpy(dtype=int),
        shape_gradients=integrator[gradient_columns]
        .to_numpy(dtype=float)
        .reshape(-1, 3, 2),
        weights=integrator["qpWeight"].to_numpy(dtype=float),
        reactions=reactions["forces"].to_numpy(dtype=float),
        reference_internal_forces=nodes[["fintx", "finty"]].to_numpy(
            dtype=float
        ),
        reference_deformation_gradient=_as_matrices(elements, "F"),
        reference_stress=_as_matrices(elements, "P"),
    )


def read_field_steps(
    directory: pathlib.Path, load_steps: tuple[int, ...]
) -> list[FieldStep]:
    """Reads the EUCLID finite element data of several load steps.

    Args:
        directory: The cache directory of the EUCLID data source.
        load_steps: The load step numbers.

    Returns:
        One field step for each load step, in the given order.
    """
    return [read_field_step(directory, step) for step in load_steps]


def homogeneous_summary(tests: list[HomogeneousTest]) -> pd.DataFrame:
    """Summarizes homogeneous tests for display.

    Args:
        tests: The homogeneous tests.

    Returns:
        A table with one row per test.
    """
    rows = [
        {
            "source": test.source_key,
            "material": test.material,
            "mode": test.mode,
            "points": test.stretch.size,
            "max stretch": float(test.stretch.max()),
            "max stress": float(test.nominal_stress.max()),
            "duration [s]": None
            if test.time is None
            else float(np.ptp(test.time)),
        }
        for test in tests
    ]
    return pd.DataFrame(rows)


def field_summary(steps: list[FieldStep]) -> pd.DataFrame:
    """Summarizes field steps for display.

    Args:
        steps: The field steps.

    Returns:
        A table with one row per load step.
    """
    rows = [
        {
            "load step": step.load_step,
            "nodes": step.coordinates.shape[0],
            "elements": step.connectivity.shape[0],
            "free degrees of freedom": int(np.sum(step.dirichlet_codes == 0)),
            "max |u|": float(np.linalg.norm(step.displacement, axis=1).max()),
            "reactions": np.round(step.reactions, 4).tolist(),
        }
        for step in steps
    ]
    return pd.DataFrame(rows).set_index("load step")
