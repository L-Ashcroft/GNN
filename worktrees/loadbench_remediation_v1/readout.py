"""Train-only local affine readout; no torch, dataset I/O or test labels."""
import numpy as np


def predict(train_x, query_x, z, observed, *, k=32, affine=True, ridge=.01):
    """Per-target observed-neighbor search; ridge leaves intercept unpenalized.

    All inputs must already use TRAIN-fitted transforms. Predictions (not labels)
    are bounded to the selected training neighbors' observed output range.
    """
    out = np.zeros((len(query_x), z.shape[1]), dtype=np.float64)
    counts = np.zeros(z.shape[1], dtype=np.int64)
    patterns = {}
    for j in range(z.shape[1]):
        key = observed[:, j].tobytes()
        patterns.setdefault(key, []).append(j)
    for slots in patterns.values():
        selected = np.flatnonzero(observed[:, slots[0]])
        if not len(selected):
            continue  # No observation is reported by the caller for these slots.
        counts[slots] = len(selected)
        a = train_x[selected]; y = z[selected][:, slots]
        kk = min(k, len(a)); norm = np.square(a).sum(1)
        for start in range(0, len(query_x), 32):
            q = query_x[start:start+32]
            distance = np.maximum(0., np.square(q).sum(1)[:, None] + norm - 2*q@a.T)
            ix = np.argpartition(distance, kk-1, axis=1)[:, :kk]
            dd = np.take_along_axis(distance, ix, axis=1)
            weights = 1/(dd+1e-4); weights /= weights.sum(1, keepdims=True)
            yy = y[ix]; center = (weights[:, :, None]*yy).sum(1)
            p = center.copy()
            if affine and kk > 1 and a.shape[1]:
                xx = a[ix]; mx = (weights[:, :, None]*xx).sum(1)
                xc = xx-mx[:, None, :]
                # Local diagonal scaling prevents a few feature dimensions
                # from setting the ridge penalty's numerical scale.
                scale = np.sqrt((weights[:, :, None]*xc**2).sum(1)).clip(1e-3)
                xc = xc/scale[:, None, :]
                bb = xc*np.sqrt(weights)[:, :, None]
                yc = (yy-center[:, None, :])*np.sqrt(weights)[:, :, None]
                dual = np.linalg.solve(bb@bb.transpose(0, 2, 1)+ridge*np.eye(kk), yc)
                beta = bb.transpose(0, 2, 1)@dual
                p += np.einsum('bd,bdj->bj', (q-mx)/scale, beta)
                p = np.minimum(np.maximum(p, yy.min(1)), yy.max(1))
            out[start:start+len(q), np.asarray(slots)] = p
    if not np.isfinite(out).all():
        raise ValueError('Non-finite local prediction')
    return out, counts


def normalize(train, query):
    center = train.mean(0); std = train.std(0); active = std > 1e-9
    return ((train[:, active]-center[active])/std[active],
            (query[:, active]-center[active])/std[active], center, std, active)
