"""Tests for the parameters, evaluation and experiment modules."""

import numpy as np
import pytest
from test_fitting import STRETCH, synthetic_test

from hyperfit import (
    data,
    evaluation,
    experiment,
    fitting,
    models,
    parameters,
    sources,
)


@pytest.mark.parametrize("source", sources.ALL_SOURCES, ids=lambda s: s.key)
def test_specs_follow_parameter_order_and_bounds(source):
    variants = (
        models.field_variants()
        if source is sources.EUCLID_PLATE_HOLE
        else models.homogeneous_variants()
    )
    for variant in variants:
        specs = parameters.parameter_specs(variant, source.key)
        assert [spec.name for spec in specs] == list(variant.parameter_names)
        lower = parameters.lower_vector(specs)
        upper = parameters.upper_vector(specs)
        initial = parameters.initial_vector(specs)
        assert np.all(lower < upper)
        assert np.all((lower <= initial) & (initial <= upper))


def test_euclid_expected_values_are_the_generator_values():
    specs = parameters.parameter_specs(
        models.yeoh_variant(2, compressible=True), sources.EUCLID_PLATE_HOLE.key
    )
    np.testing.assert_allclose(
        parameters.expected_vector(specs), [0.5, 0.0, 1.5]
    )


def test_missing_reference_gives_nan():
    specs = parameters.parameter_specs(
        models.yeoh_variant(2), sources.SRMD_ECOFLEX_0030.key
    )
    assert np.isnan(parameters.expected_vector(specs)).all()


def test_error_metrics():
    measured = np.array([1.0, 2.0, 4.0])
    assert evaluation.relative_rmse(measured, measured) == 0.0
    assert evaluation.relative_rmse(1.1 * measured, measured) == pytest.approx(
        0.1
    )
    assert evaluation.r_squared(measured, measured) == 1.0


def test_initial_shear_modulus():
    assert (
        evaluation.initial_shear_modulus(
            models.yeoh_variant(2), np.array([0.2, 0.1])
        )
        == 0.4
    )
    ogden = evaluation.initial_shear_modulus(
        models.ogden_variant(2), np.array([0.6, 2.0, 0.1, 4.0])
    )
    assert ogden == pytest.approx(0.5 * (1.2 + 0.4))


def test_yeoh_stability_margin():
    assert (
        evaluation.yeoh_stability_margin(np.array([0.2, -0.002, 0.0001]), 20.0)
        > 0.0
    )
    assert (
        evaluation.yeoh_stability_margin(np.array([0.2, -0.1, 0.0]), 20.0) < 0.0
    )


def test_tree_distance_table_is_symmetric():
    table = evaluation.tree_distance_table(models.homogeneous_variants())
    values = table.to_numpy()
    np.testing.assert_array_equal(values, values.T)
    assert np.all(np.diag(values) == 0)


def test_fit_with_all_optimizers_skips_linear_fit_for_ogden():
    variant = models.ogden_variant(2)
    test = synthetic_test(
        variant, [0.6, 1.3, 0.001, 5.0], data.UNIAXIAL, STRETCH
    )
    residual = fitting.homogeneous_residual_function(variant, [test])
    specs = parameters.parameter_specs(variant, sources.TRELOAR_1944.key)
    results = experiment.fit_with_all_optimizers(variant, residual, specs)
    assert set(results) == {experiment.LM, experiment.CMA_ES}
    assert experiment.best_result(results).cost < 1e-10
