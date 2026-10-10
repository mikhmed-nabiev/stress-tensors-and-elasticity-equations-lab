"""Registry of the remote data sources.

Each data source lists its files, their HTTP URLs and their SHA-256
checksums. Every URL points to a fixed git commit. A fixed commit keeps
the downloaded bytes stable, so the checksums stay valid.
"""

import dataclasses

import pandas as pd


@dataclasses.dataclass(frozen=True)
class RemoteFile:
    """One file that the package downloads over HTTP.

    Attributes:
        url: The HTTP URL of the file.
        sha256: The expected SHA-256 checksum of the file content.
        local_name: The file path inside the cache directory of the
            data source.
    """

    url: str
    sha256: str
    local_name: str


@dataclasses.dataclass(frozen=True)
class DataSource:
    """A published data source with its citation.

    Attributes:
        key: A short identifier. The cache directory uses this name.
        title: A human-readable title.
        citation: The reference to cite for the data.
        homepage: The web page of the data source.
        license: The license text or a short note about the license.
        files: The remote files of the data source.
    """

    key: str
    title: str
    citation: str
    homepage: str
    license: str
    files: tuple[RemoteFile, ...]


_SRMD_COMMIT = "8ff66c3bdc17bd226eabe83b778e13ea1e2ef8ce"
_SRMD_RAW = (
    "https://raw.githubusercontent.com/LucMarechal/"
    f"Soft-Robotics-Materials-Database/{_SRMD_COMMIT}/Tensile-Tests-Data"
)

SRMD_ECOFLEX_0030 = DataSource(
    key="srmd_ecoflex_00_30",
    title="Soft Robotics Materials Database: Ecoflex 00-30 uniaxial test",
    citation=(
        "Marechal, L., Balland, P., Lindenroth, L., Petrou, F., "
        "Kontovounisios, C., Bello, F. (2021). Toward a common framework "
        "and database of materials for soft robotics. Soft Robotics, "
        "8(3), 284-297. doi:10.1089/soro.2019.0115"
    ),
    homepage="https://github.com/LucMarechal/Soft-Robotics-Materials-Database",
    license="Open Database License (ODbL-1.0); contents under DbCL-1.0.",
    files=(
        RemoteFile(
            url=f"{_SRMD_RAW}/Ecoflex%2000-30.csv",
            sha256=(
                "34c00bcc55c401e2c6da596222deed71"
                "a8a51ba0be590a5aebe32eac07a19b27"
            ),
            local_name="Ecoflex_00-30.csv",
        ),
    ),
)

_CANN_COMMIT = "a639ea97acfa1064005381645c1a2fda74bcbfcd"

TRELOAR_1944 = DataSource(
    key="treloar_1944",
    title="Treloar (1944) natural rubber: uniaxial, equibiaxial, pure shear",
    citation=(
        "Treloar, L. R. G. (1944). Stress-strain data for vulcanised "
        "rubber under various types of deformation. Transactions of the "
        "Faraday Society, 40, 59-70. Digital copy: Linka, K., Kuhl, E. "
        "(2023). A new family of constitutive artificial neural networks "
        "towards automated model discovery. Computer Methods in Applied "
        "Mechanics and Engineering, 403, 115731."
    ),
    homepage="https://github.com/LivingMatterLab/CANN",
    license="No license file in the repository. Cite the original paper.",
    files=(
        RemoteFile(
            url=(
                "https://raw.githubusercontent.com/LivingMatterLab/CANN/"
                f"{_CANN_COMMIT}/GICANN/input/CANNsRUBBERdata.xlsx"
            ),
            sha256=(
                "7d2411b03f9f7939f67c0729289000eb"
                "a3704b87be3a5ca4f88813a18ec87750"
            ),
            local_name="CANNsRUBBERdata.xlsx",
        ),
    ),
)

_EUCLID_COMMIT = "e4d680c929d13e2b1d17f63f80708ded09efdca1"
_EUCLID_RAW = (
    "https://raw.githubusercontent.com/EUCLID-code/EUCLID-hyperelasticity/"
    f"{_EUCLID_COMMIT}/FEM_data/plate_hole_1k/plate_hole_1k_NeoHookeanJ2"
)

EUCLID_LOAD_STEPS = (10, 20, 30, 40)

_EUCLID_SHA256 = {
    (10, "output_nodes"): (
        "17918cb39b748c2e474b4015672e8c95a451f04b0ab3d160f54c557a27f249e7"
    ),
    (10, "output_elements"): (
        "04a379f179d676a22f5da387682950b1a6a6839a076d31a4f0f06de987e852fe"
    ),
    (10, "output_integrator"): (
        "60410f302fe81ce57cc0057f903bd6f8bbeaf3dbed4c030a67b538b5a2b048d4"
    ),
    (10, "output_reactions"): (
        "3a76ccc21861a8a73afe8aa6f11e402063b7c506fadcf8f63e387dbbd6b7993d"
    ),
    (20, "output_nodes"): (
        "18a33993b7187a8b209917a5d7da28ac713a9449349c4b6e0617fc4614921636"
    ),
    (20, "output_elements"): (
        "5d365ff73935649050b832811f51ff59bf36d54de90f862a2f42c937f8511a36"
    ),
    (20, "output_integrator"): (
        "60410f302fe81ce57cc0057f903bd6f8bbeaf3dbed4c030a67b538b5a2b048d4"
    ),
    (20, "output_reactions"): (
        "f66691fd752eca82558f49e0e9f824cf5bee3153f0257db4ecad11b5feeb9d65"
    ),
    (30, "output_nodes"): (
        "05094b0de476d3a9d5962e4ec2740f173685c01ae4229238ef2d8ee78b0290d1"
    ),
    (30, "output_elements"): (
        "e61c4df9fb25326f3b0a5da27956eb4e62eb060696a1e01ec99453873fdf9cc0"
    ),
    (30, "output_integrator"): (
        "60410f302fe81ce57cc0057f903bd6f8bbeaf3dbed4c030a67b538b5a2b048d4"
    ),
    (30, "output_reactions"): (
        "09329ab111135398a289cfc0e39667781d300dbd4519b625c753baa4357a3e47"
    ),
    (40, "output_nodes"): (
        "2aa06e25ae5f5ce125957be22bb507608b41e4fabe61948596e649b14969278f"
    ),
    (40, "output_elements"): (
        "f2010ec2067b9c342f7ecf18f18280a7ab0016bb96aee0573733b9cf3df0b04e"
    ),
    (40, "output_integrator"): (
        "60410f302fe81ce57cc0057f903bd6f8bbeaf3dbed4c030a67b538b5a2b048d4"
    ),
    (40, "output_reactions"): (
        "8276ca65496ed7114198f37a54d74f76983aa8c0aaa893c7fa9f377f4370b657"
    ),
}


def _euclid_files() -> tuple[RemoteFile, ...]:
    """Lists the remote files of the EUCLID plate-with-hole data.

    Returns:
        One remote file for each load step and each CSV table.
    """
    files = []
    for (load_step, table), checksum in _EUCLID_SHA256.items():
        files.append(
            RemoteFile(
                url=f"{_EUCLID_RAW}/{load_step}/{table}.csv",
                sha256=checksum,
                local_name=f"{load_step}/{table}.csv",
            )
        )
    return tuple(files)


EUCLID_PLATE_HOLE = DataSource(
    key="euclid_plate_hole",
    title="EUCLID finite element data: plate with a hole, 4 load steps",
    citation=(
        "Flaschel, M., Kumar, S., De Lorenzis, L. (2021). Unsupervised "
        "discovery of interpretable hyperelastic constitutive laws. "
        "Computer Methods in Applied Mechanics and Engineering, 381, "
        "113852. doi:10.1016/j.cma.2021.113852"
    ),
    homepage="https://github.com/EUCLID-code/EUCLID-hyperelasticity",
    license="GNU General Public License v3.0.",
    files=_euclid_files(),
)

ALL_SOURCES = (SRMD_ECOFLEX_0030, TRELOAR_1944, EUCLID_PLATE_HOLE)

MODEL_REFERENCES = (
    "Yeoh, O. H. (1993). Some forms of the strain energy function for "
    "rubber. Rubber Chemistry and Technology, 66(5), 754-771.",
    "Ogden, R. W. (1972). Large deformation isotropic elasticity: on the "
    "correlation of theory and experiment for incompressible rubberlike "
    "solids. Proceedings of the Royal Society A, 326(1567), 565-584.",
    "Hansen, N. (2016). The CMA evolution strategy: a tutorial. "
    "arXiv:1604.00772.",
    "Zhang, K., Shasha, D. (1989). Simple fast algorithms for the editing "
    "distance between trees and related problems. SIAM Journal on "
    "Computing, 18(6), 1245-1262.",
)


def sources_table(
    data_sources: tuple[DataSource, ...] = ALL_SOURCES,
) -> pd.DataFrame:
    """Lists the data sources for display.

    Args:
        data_sources: The data sources.

    Returns:
        A table with one row per data source.
    """
    rows = [
        {
            "key": source.key,
            "title": source.title,
            "files": len(source.files),
            "homepage": source.homepage,
            "license": source.license,
            "citation": source.citation,
        }
        for source in data_sources
    ]
    return pd.DataFrame(rows).set_index("key")
