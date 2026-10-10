"""Tests for hyperfit.data and hyperfit.sources. No test uses the network."""

import hashlib
import io
import re

import numpy as np
import pytest

from hyperfit import data, sources


def test_compute_sha256(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    assert data.compute_sha256(path) == hashlib.sha256(b"abc").hexdigest()


def test_download_file_uses_valid_cache(tmp_path, monkeypatch):
    content = b"cached content"
    remote = sources.RemoteFile(
        "https://example.org/f", hashlib.sha256(content).hexdigest(), "f.txt"
    )
    (tmp_path / "f.txt").write_bytes(content)

    def fail(*args, **kwargs):
        raise AssertionError("The cache was not used.")

    monkeypatch.setattr(data.urllib.request, "urlopen", fail)
    assert data.download_file(remote, tmp_path).read_bytes() == content


def test_download_file_checks_checksum(tmp_path, monkeypatch):
    remote = sources.RemoteFile("https://example.org/f", "0" * 64, "sub/f.txt")
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError):
        data.download_file(remote, tmp_path)
    assert list((tmp_path / "sub").iterdir()) == []


def test_download_file_writes_verified_file(tmp_path, monkeypatch):
    content = b"good content"
    remote = sources.RemoteFile(
        "https://example.org/f", hashlib.sha256(content).hexdigest(), "f.txt"
    )
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content)
    )
    assert data.download_file(remote, tmp_path).read_bytes() == content


def test_sources_are_pinned_and_complete():
    for source in sources.ALL_SOURCES:
        names = [remote.local_name for remote in source.files]
        assert len(names) == len(set(names))
        for remote in source.files:
            assert re.search(r"/[0-9a-f]{40}/", remote.url), remote.url
            assert re.fullmatch(r"[0-9a-f]{64}", remote.sha256)
    assert len(sources.EUCLID_PLATE_HOLE.files) == 4 * len(
        sources.EUCLID_LOAD_STEPS
    )


def test_srmd_parser(srmd_file):
    metadata = data.read_srmd_metadata(srmd_file)
    assert metadata["Material"] == "TestRubber"
    test = data.srmd_uniaxial_test(srmd_file, "key")
    assert test.mode == data.UNIAXIAL
    np.testing.assert_allclose(test.stretch, [1.0, 1.1, 1.2, 1.3, 1.4])
    np.testing.assert_allclose(
        test.nominal_stress, [0.0, 0.01, 0.02, 0.03, 0.024]
    )
    np.testing.assert_allclose(test.time, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_preprocessing_steps(srmd_file):
    test = data.srmd_uniaxial_test(srmd_file, "key")
    truncated = data.truncate_at_peak_stress(test)
    np.testing.assert_allclose(truncated.stretch, [1.0, 1.1, 1.2, 1.3])
    loaded = data.drop_unloaded_points(truncated)
    np.testing.assert_allclose(loaded.stretch, [1.1, 1.2, 1.3])
    np.testing.assert_allclose(loaded.time, [1.0, 2.0, 3.0])


def test_treloar_parser(treloar_file):
    tests = data.read_treloar_tests(treloar_file, "key")
    assert [test.mode for test in tests] == list(data.TEST_MODES)
    np.testing.assert_allclose(tests[0].stretch, [1.0, 1.5, 2.0])
    np.testing.assert_allclose(tests[0].nominal_stress, [0.0, 0.39, 0.59])
    np.testing.assert_allclose(tests[2].stretch, [1.0, 1.3])


def test_treloar_parser_rejects_other_layout(tmp_path, treloar_file):
    import openpyxl

    workbook = openpyxl.load_workbook(treloar_file)
    workbook["Sheet1"]["A2"] = "XX"
    broken = tmp_path / "broken.xlsx"
    workbook.save(broken)
    with pytest.raises(ValueError):
        data.read_treloar_tests(broken, "key")


def test_field_parser(euclid_directory, square_step):
    step = data.read_field_step(euclid_directory, 7)
    assert step.load_step == 7
    np.testing.assert_allclose(step.displacement, square_step.displacement)
    np.testing.assert_array_equal(
        step.dirichlet_codes, square_step.dirichlet_codes
    )
    np.testing.assert_allclose(
        step.shape_gradients, square_step.shape_gradients
    )
    np.testing.assert_allclose(step.reactions, [-1.0, 1.0, -2.0, 2.0])
    assert step.reference_deformation_gradient.shape == (4, 2, 2)


def test_rebuild_uniform_time(srmd_file):
    test = data.srmd_uniaxial_test(srmd_file, "key")
    broken = data.dataclasses.replace(
        test, time=np.array([0.0, 1.0, 2.0, 1.5, 1.6])
    )
    repaired = data.rebuild_uniform_time(broken)
    np.testing.assert_allclose(repaired.time, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(repaired.stretch, test.stretch)


def test_rebuild_uniform_time_needs_time(srmd_file):
    test = data.srmd_uniaxial_test(srmd_file, "key")
    with pytest.raises(ValueError):
        data.rebuild_uniform_time(data.dataclasses.replace(test, time=None))
