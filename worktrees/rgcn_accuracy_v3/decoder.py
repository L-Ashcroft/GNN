"""Ridge-initialized nonlinear physical/graph skip decoder, masked Huber."""
import numpy as np


class Decoder:
    def __init__(self, x, target, mask, seed=42, hidden=64, delta=2.):
        self.target = target; self.mask = mask; self.delta = delta
        self.rng = np.random.default_rng(seed)
        d, k = x.shape[1], target.shape[1]
        w = np.zeros((d, k)); b = np.zeros(k)
        for j in range(k):
            keep = mask[:, j]; a = x[keep].astype('float64'); y = target[keep, j]
            if not len(a): continue
            mx, my = a.mean(0), y.mean(); a -= mx; y = y - my
            if len(a) < d:
                w[:, j] = a.T @ np.linalg.solve(a @ a.T + 10. * np.eye(len(a)), y)
            else:
                w[:, j] = np.linalg.solve(a.T @ a + 10. * np.eye(d), a.T @ y)
            b[j] = my - mx @ w[:, j]
        bound = np.sqrt(6. / (d + hidden))
        self.p = dict(w=w, b=b, h=self.rng.uniform(-bound, bound, (d, hidden)),
                      hb=np.zeros(hidden), o=np.zeros((hidden, k)), ob=np.zeros(k))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}; self.step = 0

    def forward(self, x):
        p = self.p; u = x @ p['h'] + p['hb']; sig = np.exp(-np.logaddexp(0., -u)); a = u * sig
        return x @ p['w'] + p['b'] + a @ p['o'] + p['ob'], (u, sig, a)

    def gradient(self, x, indices):
        pred, (u, sig, a) = self.forward(x); mask = self.mask[indices]
        error = np.where(mask, pred - self.target[indices], 0.)
        count = max(1, int(mask.sum())); absolute = abs(error)
        loss = np.where(absolute <= self.delta, .5 * error**2,
                        self.delta * (absolute - .5 * self.delta)).sum() / count
        delta = np.clip(error, -self.delta, self.delta) / count
        dh = (delta @ self.p['o'].T) * (sig + u * sig * (1. - sig))
        grad = dict(w=x.T @ delta, b=delta.sum(0), h=x.T @ dh, hb=dh.sum(0), o=a.T @ delta, ob=delta.sum(0))
        return float(loss), grad

    def update(self, grad, factor=1.):
        self.step += 1
        norm = np.sqrt(sum(float(np.square(g).sum()) for g in grad.values()))
        clip = min(1., 5. / max(norm, 1e-12))
        for key, p in self.p.items():
            g = grad[key] * clip
            self.m[key] = .9 * self.m[key] + .1 * g
            self.v[key] = .999 * self.v[key] + .001 * g**2
            lr = (1e-4 if key in ['w', 'b'] else 1e-3) * factor
            p *= 1. - lr * .01
            p -= lr * (self.m[key] / (1. - .9**self.step)) / (np.sqrt(self.v[key] / (1. - .999**self.step)) + 1e-8)
            assert np.isfinite(p).all()

    def state(self): return {k: v.copy() for k, v in self.p.items()}


def transform(y, scales, kinds):
    z = np.asarray(y, dtype='float64') / scales
    out = np.sign(z) * np.log1p(abs(z))
    for j, kind in enumerate(kinds):
        if kind == 'linear': out[:, j] = z[:, j]
    return out


def physical(z, scales, kinds):
    out = z.copy()
    for j, kind in enumerate(kinds):
        if kind == 'symlog': out[:, j] = np.sign(z[:, j]) * np.expm1(abs(z[:, j]))
    return out * scales
