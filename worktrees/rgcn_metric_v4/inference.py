"""Frozen RGCN feature + supervised-neighborhood experts, known-topology ID."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'worktrees/rgcn_accuracy_v3'))
from common import read, sha
from decoder import physical
sys.path.insert(0, str(ROOT / 'worktrees/loadbench_remediation_v1'))
from readout import predict
import importlib.util
_spec = importlib.util.spec_from_file_location('metric_v4_training_features', Path(__file__).with_name('train.py'))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
feature_modes = _module.feature_modes


class MetricReadout:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else ROOT / 'artifacts/rgcn_metric_v4'
        self.recipe = read(self.directory / 'recipe.json'); self.cache = {}

    def bank(self, topology):
        if topology in self.cache: return self.cache[topology]
        self.cache.clear()
        folder = self.directory / topology; selected = read(folder / 'selection_frozen.json')
        assert sha(folder / 'model.npz') == selected['model_sha256']
        assert sha(folder / 'normalization.json') == selected['normalization_sha256']
        assert selected['recipe_sha256'] == sha(self.directory / 'recipe.json')
        with np.load(folder / 'model.npz', allow_pickle=False) as f: state = {k: f[k].copy() for k in f.files}
        value = state, read(folder / 'normalization.json'), selected
        self.cache[topology] = value; return value

    def __call__(self, topology, parameters, embedding, baseline_physical, *, parameter_names):
        state, norm, selected = self.bank(topology)
        if list(parameter_names) != norm['parameter_names']: raise ValueError('Canonical parameter names/order differs')
        params = np.asarray(parameters, dtype='float64'); e = np.asarray(embedding, dtype='float64')
        base = np.asarray(baseline_physical, dtype='float64')
        assert params.ndim == 2 and params.shape[1] == len(parameter_names) and e.shape == (len(params), 128) and base.shape == (len(params), 16)
        assert np.isfinite(params).all() and np.isfinite(e).all() and np.isfinite(base).all()
        units = np.array(norm['parameter_scales']); params = np.sign(params) * np.log1p(abs(params) / units)
        pa = state['parameter_active']; ea = state['embedding_active']
        q = (params[:, pa] - state['parameter_center'][pa]) / state['parameter_std'][pa]
        ge = (e[:, ea] - state['embedding_center'][ea]) / state['embedding_std'][ea]
        txp, te = state['train_parameter'], state['train_embedding']
        tw, tb = state['projection_weight'], state['projection_bias']
        train_features = feature_modes(txp, te, np.column_stack([txp, te]) @ tw + tb)
        query_features = feature_modes(q, ge, np.column_stack([q, ge]) @ tw + tb)
        groups = {}
        for slot, choice in selected['choices'].items():
            if choice['mode'] != 'identity': groups.setdefault((choice['mode'], choice['neighbors']), []).append(int(slot))
        result = base.copy()
        for (mode, neighbors), slots in groups.items():
            z, counts = predict(train_features[mode], query_features[mode], state['train_z'][:, slots], state['train_mask'][:, slots], k=neighbors, ridge=.01)
            assert (counts > 0).all()
            candidate = physical(z, state['scales'][slots], [norm['kinds'][j] for j in slots])
            for i, slot in enumerate(slots):
                weight = selected['choices'][str(slot)]['weight']
                result[:, slot] = (1. - weight) * base[:, slot] + weight * candidate[:, i]
        assert np.isfinite(result).all()
        return result
