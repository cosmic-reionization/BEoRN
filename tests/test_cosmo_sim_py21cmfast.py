"""Unit tests for Py21cmFastLoader's py21cmfast version detection and
v3/v4 API dispatch. These never import the real py21cmfast package -- a
fake module is injected into sys.modules so the dispatch logic (and its
logging) can be exercised without the heavy compiled dependency installed."""
import itertools
import json
import logging
import sys
import types
from pathlib import Path

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


# ---------------------------------------------------------------------------
# py21cmfast_cache_direc resolution
# ---------------------------------------------------------------------------

def test_resolve_py21cmfast_cache_direc_defaults_true():
    loader = Py21cmFastLoader(Parameters())
    assert loader.py21cmfast_cache_direc is True


def test_resolve_py21cmfast_cache_direc_true_resolves_under_file_root(tmp_path):
    loader = Py21cmFastLoader(Parameters())
    assert loader._resolve_py21cmfast_cache_direc(tmp_path) == tmp_path / '_py21cmfast_cache'


@pytest.mark.parametrize('setting', [False, None])
def test_resolve_py21cmfast_cache_direc_disabled(tmp_path, setting):
    loader = Py21cmFastLoader(Parameters(), py21cmfast_cache_direc=setting)
    assert loader._resolve_py21cmfast_cache_direc(tmp_path) is None


def test_resolve_py21cmfast_cache_direc_explicit_path(tmp_path):
    custom = tmp_path / 'somewhere_else'
    loader = Py21cmFastLoader(Parameters(), py21cmfast_cache_direc=custom)
    assert loader._resolve_py21cmfast_cache_direc(tmp_path) == custom


# ---------------------------------------------------------------------------
# generate() <-> _generate_v4 dispatch: chained/resume flags, chain manifest
# ---------------------------------------------------------------------------

def test_generate_dispatches_chained_true_by_default(tmp_path, monkeypatch):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('4.2.0'))
    captured = {}
    monkeypatch.setattr(
        loader, '_generate_v4',
        lambda p21c, file_root, missing, chained, resume: captured.update(chained=chained, resume=resume),
    )

    loader.generate(handler)

    assert captured == {'chained': True, 'resume': False}


def test_generate_dispatches_chained_false_when_disabled(tmp_path, monkeypatch):
    loader, handler = _loader_and_handler(tmp_path)
    loader.parameters.halo_sim.chain_halos = False
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('4.2.0'))
    captured = {}
    monkeypatch.setattr(
        loader, '_generate_v4',
        lambda p21c, file_root, missing, chained, resume: captured.update(chained=chained),
    )

    loader.generate(handler)

    assert captured == {'chained': False}


def test_generate_writes_chain_manifest_matching_full_redshift_list(tmp_path, monkeypatch):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('4.2.0'))
    monkeypatch.setattr(loader, '_generate_v4', lambda *a, **k: None)

    loader.generate(handler)

    manifest = loader.file_root / 'chain_redshifts.json'
    assert manifest.exists()
    expected = sorted((float(z) for z in loader.redshifts), reverse=True)
    assert json.loads(manifest.read_text()) == expected


def test_generate_warns_and_forces_full_regen_when_redshift_list_changes(tmp_path, monkeypatch, caplog):
    params = Parameters()
    params.cosmo_sim.snapshot_redshifts = np.array([10.0, 9.0])
    loader = Py21cmFastLoader(params)
    handler = Handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('4.2.0'))
    monkeypatch.setattr(loader, '_generate_v4', lambda *a, **k: None)

    loader.generate(handler)  # first, uneventful run -- writes the manifest for [10.0, 9.0]

    # Pretend z=10 already finished (its own output files exist on disk).
    file_root = loader.file_root
    (file_root / 'haloes_z10.000.h5').touch()
    (file_root / 'densities_z10.000.h5').touch()

    # Now change the redshift list.
    params.cosmo_sim.snapshot_redshifts = np.array([10.0, 9.5, 9.0])
    captured = {}
    monkeypatch.setattr(
        loader, '_generate_v4',
        lambda p21c, file_root, missing, chained, resume: captured.update(missing=missing, resume=resume),
    )

    with caplog.at_level(logging.WARNING):
        loader.generate(handler)

    assert any('redshift list' in r.message.lower() for r in caplog.records)
    assert sorted(captured['missing']) == [9.0, 9.5, 10.0]
    assert captured['resume'] is False


def test_generate_no_warning_when_redshift_list_unchanged(tmp_path, monkeypatch, caplog):
    loader, handler = _loader_and_handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_py21cmfast('4.2.0'))
    monkeypatch.setattr(loader, '_generate_v4', lambda *a, **k: None)
    loader.generate(handler)

    with caplog.at_level(logging.WARNING):
        loader.generate(handler)

    assert not any('redshift list' in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# _generate_v4 internals: descendant-chaining order, native-cache kwargs,
# and the completion log message -- exercised against a small fake
# py21cmfast module that implements just enough of the v4 API.
# ---------------------------------------------------------------------------

class _FakeCatalog:
    _ids = itertools.count()

    def __init__(self, redshift, descendant_halos):
        self.redshift = redshift
        self.descendant_halos = descendant_halos
        self._id = next(_FakeCatalog._ids)

    def get(self, name):
        if name == 'halo_masses':
            return np.array([1e9, 2e9]) * (self._id + 1)
        if name == 'halo_coords':
            return np.zeros((2, 3))
        raise KeyError(name)


class _FakePerturbedField:
    def get(self, name):
        if name == 'density':
            return np.zeros((2, 2, 2))
        raise KeyError(name)


class _FakeOutputCache:
    def __init__(self, direc):
        self.direc = Path(direc)
        self.direc.mkdir(parents=True, exist_ok=True)


def _fake_p21c_v4_api(version='4.2.0'):
    """A fake py21cmfast v4 module recording every determine_halo_catalog
    call's kwargs, so chaining order/descendant threading/cache kwargs can
    be asserted on without the real (heavy, compiled) dependency."""
    module = types.ModuleType('py21cmfast')
    module.__version__ = version
    module.CosmoParams = lambda **kw: kw
    module.SimulationOptions = lambda **kw: kw
    module.MatterOptions = lambda **kw: kw
    module.AstroParams = lambda **kw: kw
    module.AstroOptions = lambda **kw: kw
    module.InputParameters = lambda **kw: kw
    module.OutputCache = _FakeOutputCache

    calls = {'determine_halo_catalog': [], 'perturb_field': [], 'compute_initial_conditions': []}

    def compute_initial_conditions(**kw):
        calls['compute_initial_conditions'].append(kw)
        return object()

    def perturb_field(**kw):
        calls['perturb_field'].append(kw)
        return _FakePerturbedField()

    def determine_halo_catalog(**kw):
        calls['determine_halo_catalog'].append(kw)
        return _FakeCatalog(kw['redshift'], kw.get('descendant_halos'))

    def perturb_halo_catalog(**kw):
        return kw['halo_catalog']

    module.compute_initial_conditions = compute_initial_conditions
    module.perturb_field = perturb_field
    module.determine_halo_catalog = determine_halo_catalog
    module.perturb_halo_catalog = perturb_halo_catalog
    module._test_calls = calls
    return module


def test_generate_v4_chained_visits_ascending_and_threads_descendant(tmp_path, monkeypatch):
    params = Parameters()
    params.cosmo_sim.snapshot_redshifts = np.array([12.0, 10.0, 8.0])  # descending, as typical
    loader = Py21cmFastLoader(params, py21cmfast_cache_direc=False)
    handler = Handler(tmp_path)
    fake = _fake_p21c_v4_api()
    monkeypatch.setitem(sys.modules, 'py21cmfast', fake)

    loader.generate(handler)

    calls = fake._test_calls['determine_halo_catalog']
    assert [c['redshift'] for c in calls] == [8.0, 10.0, 12.0]  # ascending: lowest z first
    assert calls[0]['descendant_halos'] is None
    assert calls[1]['descendant_halos'].redshift == 8.0
    assert calls[2]['descendant_halos'].redshift == 10.0

    for z in [12.0, 10.0, 8.0]:
        assert (loader.file_root / f'haloes_z{z:.3f}.h5').exists()
        assert (loader.file_root / f'densities_z{z:.3f}.h5').exists()


def test_generate_v4_independent_mode_never_threads_descendant(tmp_path, monkeypatch):
    params = Parameters()
    params.halo_sim.chain_halos = False
    params.cosmo_sim.snapshot_redshifts = np.array([12.0, 10.0, 8.0])
    loader = Py21cmFastLoader(params, py21cmfast_cache_direc=False)
    handler = Handler(tmp_path)
    fake = _fake_p21c_v4_api()
    monkeypatch.setitem(sys.modules, 'py21cmfast', fake)

    loader.generate(handler)

    calls = fake._test_calls['determine_halo_catalog']
    assert len(calls) == 3
    assert all(c['descendant_halos'] is None for c in calls)


def test_generate_v4_native_cache_trusted_only_when_resuming_unchanged_list(tmp_path, monkeypatch):
    params = Parameters()
    params.cosmo_sim.snapshot_redshifts = np.array([10.0, 9.0])
    loader = Py21cmFastLoader(params)  # py21cmfast_cache_direc=True (default)
    handler = Handler(tmp_path)
    fake = _fake_p21c_v4_api()
    monkeypatch.setitem(sys.modules, 'py21cmfast', fake)

    loader.generate(handler)  # first, full run
    first_calls = list(fake._test_calls['determine_halo_catalog'])
    assert first_calls  # sanity: something was actually generated
    assert all(c['write'] is True for c in first_calls)
    assert all(c['regenerate'] is True for c in first_calls)  # nothing to resume from yet

    # Simulate an interrupted second attempt: the highest redshift's own
    # output is missing, list unchanged.
    (loader.file_root / 'haloes_z10.000.h5').unlink()
    (loader.file_root / 'densities_z10.000.h5').unlink()
    fake._test_calls['determine_halo_catalog'].clear()

    loader.generate(handler)
    second_calls = fake._test_calls['determine_halo_catalog']
    assert second_calls
    assert all(c['regenerate'] is False for c in second_calls)  # unchanged list -> trust native cache


def test_generate_v4_without_native_cache_never_trusts_cache(tmp_path, monkeypatch):
    params = Parameters()
    params.cosmo_sim.snapshot_redshifts = np.array([10.0, 9.0])
    loader = Py21cmFastLoader(params, py21cmfast_cache_direc=False)
    handler = Handler(tmp_path)
    fake = _fake_p21c_v4_api()
    monkeypatch.setitem(sys.modules, 'py21cmfast', fake)

    loader.generate(handler)
    (loader.file_root / 'haloes_z10.000.h5').unlink()
    (loader.file_root / 'densities_z10.000.h5').unlink()
    fake._test_calls['determine_halo_catalog'].clear()

    loader.generate(handler)
    calls = fake._test_calls['determine_halo_catalog']
    assert calls
    assert all(c['cache'] is None for c in calls)
    assert all(c['write'] is False for c in calls)
    assert all(c['regenerate'] is True for c in calls)


def test_generate_v4_logs_cache_cleanup_message_when_cache_enabled(tmp_path, monkeypatch, caplog):
    params = Parameters()
    params.cosmo_sim.snapshot_redshifts = np.array([10.0])
    loader = Py21cmFastLoader(params)
    handler = Handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_p21c_v4_api())

    with caplog.at_level(logging.INFO):
        loader.generate(handler)

    assert any('safe to delete' in r.message.lower() for r in caplog.records)


def test_generate_v4_no_cleanup_message_when_cache_disabled(tmp_path, monkeypatch, caplog):
    params = Parameters()
    params.cosmo_sim.snapshot_redshifts = np.array([10.0])
    loader = Py21cmFastLoader(params, py21cmfast_cache_direc=False)
    handler = Handler(tmp_path)
    monkeypatch.setitem(sys.modules, 'py21cmfast', _fake_p21c_v4_api())

    with caplog.at_level(logging.INFO):
        loader.generate(handler)

    assert not any('safe to delete' in r.message.lower() for r in caplog.records)
