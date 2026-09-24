"""Orbit reading must tolerate names `opera_utils` cannot parse."""

from __future__ import annotations

import pytest
from opera_utils import CslcParseError

from disp_s1 import _baselines


def test_parsable_name_uses_get_cslc_orbit(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(_baselines, "get_cslc_orbit", lambda _f: sentinel)

    assert _baselines._get_orbit("whatever.h5") is sentinel


def test_unparsable_name_falls_back_to_the_s1_reader(monkeypatch):
    """Regression: a compressed SLC written from COMPASS-named inputs.

    It is called `compressed_<burst>_<ref>_<start>_<end>.h5`, which matches
    none of the OPERA patterns, yet carries a complete `metadata/orbit` group.
    Ministack 1 takes ministack 0's compressed SLC as its reference, so
    without the fallback every run after the first fails.
    """
    import numpy as np

    def _raise(_f):
        msg = "Unable to parse compressed_t095_000004_iw1_20170616_20161019_20170616.h5"
        raise CslcParseError(msg)

    epoch = __import__("datetime").datetime(2017, 6, 16)
    monkeypatch.setattr(_baselines, "get_cslc_orbit", _raise)
    monkeypatch.setattr(
        _baselines,
        "get_s1_orbit",
        lambda _f: (
            np.array([0.0, 10.0]),
            np.array([[7e6, 0.0, 0.0], [7e6, 1e3, 0.0]]),
            np.array([[0.0, 7e3, 0.0], [0.0, 7e3, 1.0]]),
            epoch,
        ),
    )

    orbit = _baselines._get_orbit("compressed_t095_000004_iw1_20170616_x_y.h5")

    assert orbit.size == 2


def test_other_parse_errors_are_not_swallowed(monkeypatch):
    """Only `CslcParseError` is a naming problem; nothing else is."""

    def _raise(_f):
        msg = "file is missing metadata/orbit"
        raise KeyError(msg)

    monkeypatch.setattr(_baselines, "get_cslc_orbit", _raise)

    with pytest.raises(KeyError):
        _baselines._get_orbit("x.h5")
