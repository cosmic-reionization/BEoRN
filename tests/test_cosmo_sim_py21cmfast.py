"""Unit tests for Py21cmFastLoader's py21cmfast version detection and
v3/v4 API dispatch. These never import the real py21cmfast package -- a
fake module is injected into sys.modules so the dispatch logic (and its
logging) can be exercised without the heavy compiled dependency installed."""
import logging
import sys
import types

import numpy as np
import pytest

from beorn.io import Handler
from beorn.structs import Parameters
from beorn.load_input_data.cosmo_sim_py21cmfast import (
    Py21cmFastLoader,
    _py21cmfast_major_version,
)


@pytest.mark.parametrize('version, expected', [
    ('3.4.0', 3),
    ('3.1.5', 3),
    ('4.0.0', 4),
    ('4.2.0', 4),
    ('4.2.0.dev0+g2cb6000d6.d20260613', 4),
])
def test_py21cmfast_major_version_parsing(version, expected):
    assert _py21cmfast_major_version(version) == expected


def _fake_py21cmfast(version):
    module = types.ModuleType('py21cmfast')
    module.__version__ = version
    return module


def _loader_and_handler(tmp_path):
    """A loader with exactly one, not-yet-cached snapshot redshift, so
    generate() reaches the py21cmfast import/dispatch instead of returning
    early on the "already cached" fast path."""
    params = Parameters()
    params.cosmo_sim.snapshot_redshifts = np.array([params.solver.redshifts[0]])
    return Py21cmFastLoader(params), Handler(tmp_path)


def test_generate_dispatches_to_v3_for_major_version_3(tmp_path, monkeypatch):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('3.4.0'))
    calls = []
    monkeypatch.setattr(loader, '_generate_v3', lambda *a, **k: calls.append('v3'))
    monkeypatch.setattr(loader, '_generate_v4', lambda *a, **k: calls.append('v4'))

    loader.generate(handler)

    assert calls == ['v3']


def test_generate_dispatches_to_v4_for_major_version_4(tmp_path, monkeypatch):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('4.2.0'))
    calls = []
    monkeypatch.setattr(loader, '_generate_v3', lambda *a, **k: calls.append('v3'))
    monkeypatch.setattr(loader, '_generate_v4', lambda *a, **k: calls.append('v4'))

    loader.generate(handler)

    assert calls == ['v4']


def test_generate_raises_for_unsupported_major_version(tmp_path, monkeypatch):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('5.0.0'))

    with pytest.raises(RuntimeError, match='Unsupported py21cmfast version'):
        loader.generate(handler)


def test_generate_logs_v4_upgrade_suggestion_only_for_v3(tmp_path, monkeypatch, caplog):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('3.4.0'))
    monkeypatch.setattr(loader, '_generate_v3', lambda *a, **k: None)

    with caplog.at_level(logging.INFO):
        loader.generate(handler)

    assert any('consider upgrading' in r.message.lower() for r in caplog.records)


def test_generate_does_not_suggest_upgrade_for_v4(tmp_path, monkeypatch, caplog):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('4.2.0'))
    monkeypatch.setattr(loader, '_generate_v4', lambda *a, **k: None)

    with caplog.at_level(logging.INFO):
        loader.generate(handler)

    assert not any('consider upgrading' in r.message.lower() for r in caplog.records)


def test_generate_sets_file_root_after_dispatch(tmp_path, monkeypatch):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('4.2.0'))
    monkeypatch.setattr(loader, '_generate_v4', lambda *a, **k: None)
    assert loader.file_root is None

    loader.generate(handler)

    assert loader.file_root == handler.file_root / loader.input_tag
