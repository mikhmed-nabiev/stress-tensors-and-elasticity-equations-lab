"""Parameter vectors, bounds and expected values.

The parameter vector theta of a model variant lists the parameters in the
order of `ModelVariant.parameter_names`. This module gives, for each data
source and each variant:

* the expected value of each parameter, with its reference, if a
  reference exists;
* bounds for the bounded optimizers;
* a scale for CMA-ES;
* a start point for the iterative optimizers.

The stress scale of a data source is the typical stress magnitude of the
material. Higher Yeoh orders multiply larger powers of (I1b - 3), so their
scale is smaller by a factor of 10 per order.
"""

import dataclasses

import numpy as np
import pandas as pd

from hyperfit import models, sources

STRESS_SCALE = {
    sources.SRMD_ECOFLEX_0030.key: 0.03,
    sources.TRELOAR_1944.key: 0.3,
    sources.EUCLID_PLATE_HOLE.key: 1.0,
}
STRESS_UNIT = {
    sources.SRMD_ECOFLEX_0030.key: "MPa",
    sources.TRELOAR_1944.key: "MPa",
    sources.EUCLID_PLATE_HOLE.key: "-",
}

_MARECHAL = "Marechal et al. 2021, Table III (true stress fit)"
_OGDEN_1972 = "Ogden 1972, two leading terms of the 3-term fit"
_TRELOAR_SHEAR = "mu0 / 2, mu0 = 0.4225 MPa from the Ogden 1972 fit"
_EUCLID_TRUTH = "Data generator of Flaschel et al. 2021 (exact)"

# Published values: (source key, variant name) -> name -> (value, reference).
_PUBLISHED: dict[tuple[str, str], dict[str, tuple[float, str]]] = {
    (sources.SRMD_ECOFLEX_0030.key, "yeoh1"): {
        "C10": (0.03245 / 2.0, "Marechal et al. 2021, Table III: mu / 2"),
    },
    (sources.SRMD_ECOFLEX_0030.key, "yeoh3"): {
        "C10": (1.00e-1, _MARECHAL),
        "C20": (1.20e-2, _MARECHAL),
        "C30": (4.96e-5, _MARECHAL),
    },
    (sources.TRELOAR_1944.key, "yeoh1"): {"C10": (0.21125, _TRELOAR_SHEAR)},
    (sources.TRELOAR_1944.key, "yeoh2"): {"C10": (0.21125, _TRELOAR_SHEAR)},
    (sources.TRELOAR_1944.key, "yeoh3"): {"C10": (0.21125, _TRELOAR_SHEAR)},
    (sources.TRELOAR_1944.key, "ogden2"): {
        "mu1": (0.63, _OGDEN_1972),
        "alpha1": (1.3, _OGDEN_1972),
        "mu2": (0.0012, _OGDEN_1972),
        "alpha2": (5.0, _OGDEN_1972),
    },
}
for _order in (1, 2, 3):
    _values = {"C10": (0.5, _EUCLID_TRUTH), "D1": (1.5, _EUCLID_TRUTH)}
    for _higher in range(2, _order + 1):
        _values[f"C{_higher}0"] = (0.0, _EUCLID_TRUTH)
    _PUBLISHED[
        (sources.EUCLID_PLATE_HOLE.key, f"yeoh{_order}_compressible")
    ] = _values


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
    """The description of one parameter.

    Attributes:
        name: The parameter name.
        expected: The expected value, or None if no reference exists.
        reference: The source of the expected value.
        lower: The lower bound.
        upper: The upper bound.
        scale: The typical magnitude. CMA-ES uses it as the step size unit.
        initial: The start point for the iterative optimizers.
        unit: The unit of the parameter.
    """

    name: str
    expected: float | None
    reference: str
    lower: float
    upper: float
    scale: float
    initial: float
    unit: str


def _default_spec(name: str, stress_scale: float, unit: str) -> ParameterSpec:
    """Builds the bounds, scale and start point of one parameter.

    Args:
        name: The parameter name.
        stress_scale: The typical stress of the data source.
        unit: The stress unit of the data source.

    Returns:
        A specification without an expected value.

    Raises:
        ValueError: The parameter name is unknown.
    """
    if name.startswith("C") and name.endswith("0"):
        order = int(name[1:-1])
        scale = stress_scale / 10 ** (order - 1)
        lower = 0.0 if order == 1 else -100.0 * scale
        initial = scale if order == 1 else 0.0
        return ParameterSpec(
            name, None, "", lower, 100.0 * scale, scale, initial, unit
        )
    if name == "D1":
        return ParameterSpec(
            name,
            None,
            "",
            0.0,
            100.0 * stress_scale,
            stress_scale,
            stress_scale,
            unit,
        )
    if name.startswith("mu"):
        term = int(name[2:])
        initial = stress_scale if term == 1 else 0.01 * stress_scale
        return ParameterSpec(
            name,
            None,
            "",
            0.0,
            100.0 * stress_scale,
            stress_scale,
            initial,
            unit,
        )
    if name.startswith("alpha"):
        term = int(name[5:])
        initial = 2.0 if term == 1 else 5.0
        return ParameterSpec(name, None, "", 0.1, 20.0, 1.0, initial, "-")
    raise ValueError(f"Unknown parameter name: {name}.")


def parameter_specs(
    variant: models.ModelVariant, source_key: str
) -> list[ParameterSpec]:
    """Lists the parameter specifications of a variant for a data source.

    A published value becomes the expected value and the start point.

    Args:
        variant: The model variant.
        source_key: The key of the data source.

    Returns:
        One specification per parameter, in the order of the parameter
        vector.
    """
    published = _PUBLISHED.get((source_key, variant.name), {})
    specs = []
    for name in variant.parameter_names:
        spec = _default_spec(
            name, STRESS_SCALE[source_key], STRESS_UNIT[source_key]
        )
        if name in published:
            value, reference = published[name]
            spec = dataclasses.replace(
                spec, expected=value, reference=reference, initial=value
            )
        specs.append(spec)
    return specs


def expected_vector(specs: list[ParameterSpec]) -> np.ndarray:
    """Returns the expected values. NaN marks a missing reference."""
    return np.array(
        [np.nan if s.expected is None else s.expected for s in specs]
    )


def initial_vector(specs: list[ParameterSpec]) -> np.ndarray:
    """Returns the start point of the iterative optimizers."""
    return np.array([s.initial for s in specs])


def lower_vector(specs: list[ParameterSpec]) -> np.ndarray:
    """Returns the lower bounds."""
    return np.array([s.lower for s in specs])


def upper_vector(specs: list[ParameterSpec]) -> np.ndarray:
    """Returns the upper bounds."""
    return np.array([s.upper for s in specs])


def scale_vector(specs: list[ParameterSpec]) -> np.ndarray:
    """Returns the typical magnitudes."""
    return np.array([s.scale for s in specs])


def specs_table(specs: list[ParameterSpec]) -> pd.DataFrame:
    """Converts parameter specifications into a table for display.

    Args:
        specs: The parameter specifications.

    Returns:
        A table with one row per parameter.
    """
    table = pd.DataFrame([dataclasses.asdict(spec) for spec in specs])
    table["expected"] = table["expected"].astype(float)
    return table.set_index("name")


def specs_overview(
    variants: list[models.ModelVariant], source_key: str
) -> pd.DataFrame:
    """Lists the parameter specifications of several variants.

    Args:
        variants: The model variants.
        source_key: The key of the data source.

    Returns:
        A table with one row per variant and parameter.
    """
    tables = []
    for variant in variants:
        table = specs_table(parameter_specs(variant, source_key))
        table.insert(0, "variant", variant.title)
        tables.append(table)
    return pd.concat(tables).set_index("variant", append=True)
