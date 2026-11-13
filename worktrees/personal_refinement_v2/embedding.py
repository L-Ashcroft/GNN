"""Frozen parent GNN feature extraction, train/adaptation + validation only."""
import gc
import numpy as np
from common import ROOT, CACHE, NEW_CACHE, PARENT_OUTPUT, read, atomic, sha, now, setup_model_runtime
from features import Inputs


def prepare(parent, budget, update=lambda **kw: None):
    import psutil
    if psutil.virtual_memory().available < 1.6 * 2**30:
        return False
    data_module, model_module = setup_model_runtime()
    import torch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    inputs = Inputs(budget); inputs.parent_check(parent)
    folder = NEW_CACHE / parent; folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = PARENT_OUTPUT / parent / 'best.pt'
    source_hash = sha(checkpoint_path)
    if (folder / 'manifest.json').exists():
        saved = read(folder / 'manifest.json')
        assert saved['parent_checkpoint_sha256'] == source_hash
        assert saved['training_ids_sha256'] == inputs.training_id_hash_cache_order
        return True
    selection = read(PARENT_OUTPUT / parent / 'selection_frozen.json')
    assert selection['checkpoint_sha256'] == source_hash
    reader = data_module.Reader(verify=True)
    stats = read(PARENT_OUTPUT / parent / 'normalization.json')
    model = model_module.TargetRGCN(reader.schema, reader.tasks, stats)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    del checkpoint; gc.collect()
    maps = {}
    for name in inputs.structures:
        location = folder / name; location.mkdir(exist_ok=True)
        value = np.lib.format.open_memmap(location / 'embedding.npy', mode='w+', dtype='float32', shape=(2000,128))
        value[:] = np.nan; maps[name] = value
    refs = np.concatenate([reader.training_refs(budget), reader.refs['validation']])
    with torch.inference_mode():
        for start in range(0, len(refs), 16):
            if psutil.virtual_memory().available < 512 * 2**20:
                return False
            chosen = refs[start:start+16]
            roles = {reader.index[int(i)]['role'] for i in chosen}
            # Boundary batches may contain train + validation: keep roles explicit.
            for role in roles:
                local_refs = [i for i in chosen if reader.index[int(i)]['role'] == role]
                assert role in ['train','validation']
                batch = reader.batch(local_refs, role)
                state = model(batch, return_states=True)
                value = state['embedding'].cpu().numpy()
                for i, ref in enumerate(local_refs):
                    item = reader.index[int(ref)]; maps[item['topology_id']][item['local']] = value[i]
                del batch, state, value
            if start % 256 == 0:
                update(state='extracting_frozen_GNN', parent=parent, samples_done=start+len(chosen), role_samples=len(refs))
    for value in maps.values(): value.flush(); value._mmap.close()
    paths = [folder/name/'embedding.npy' for name in inputs.structures]
    atomic(folder / 'manifest.json', {'parent': parent, 'parent_checkpoint_sha256': source_hash,
        'training_ids_sha256': inputs.training_id_hash_cache_order, 'budget_per_topology': budget,
        'role_policy': 'Declared adaptation train rows and validation only; test embeddings remain NaN',
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, 'at': now()})
    return True
