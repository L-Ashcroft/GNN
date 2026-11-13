"""Versioned AC screening; never disguise rejected measurements with a mask."""
import hashlib
import io
import json
import zipfile
import numpy as np
from .common import ROOT,digest

POLICY={'version':'ac_continuity_v1','gain_step_dB':20.,'minimum_large_steps':10,
    'minimum_descending_crossings':20,'grid_points_per_decade':100,
    'action':'quarantine entire sample; retain original raw archive; sample another design to fill quota',
    'scope':'Amplifier AC numerical-continuity screen; not a product-specification filter'}


def ac_diagnostics(ac):
    if ac.ndim!=2 or ac.shape[1]<3 or not np.isfinite(ac).all() or np.any(np.diff(ac[:,0])<=0):
        raise ValueError('Invalid AC waveform')
    with np.errstate(divide='ignore',invalid='ignore'):
        gain=20*np.log10(np.hypot(ac[:,1],ac[:,2]))
    finite=bool(np.isfinite(gain).all())
    jumps=int(np.count_nonzero(np.abs(np.diff(gain))>POLICY['gain_step_dB']))
    crossings=int(np.count_nonzero((gain[:-1]>=0)&(gain[1:]<0)))
    return {'gain_jumps_over_20dB_per_grid_step':jumps,'descending_unity_crossings':crossings,
        'review_required':not finite or jumps>=POLICY['minimum_large_steps'] or crossings>=POLICY['minimum_descending_crossings']}


def apply_quality(entry,result):
    """Ledger gate for both fresh results and orphan archives from older runners."""
    if not result['accepted']:return result
    diagnostic=None
    if entry['family']=='Amplifier':
        with zipfile.ZipFile(ROOT/result['archive']) as z:
            wave=z.read('ac_wave.dat')
            if hashlib.sha256(wave).hexdigest()!=result['artifacts_sha256']['ac_wave.dat']:
                raise ValueError('AC artifact SHA mismatch')
        diagnostic=ac_diagnostics(np.loadtxt(io.BytesIO(wave),ndmin=2))
    revised={**result,'quality_policy_version':POLICY['version'],'quality_policy_sha256':digest(POLICY),
        'ac_quality_diagnostics':diagnostic}
    if diagnostic and diagnostic['review_required']:
        revised.update(accepted=False,failure='QualityReview: AC numerical-continuity screen; original raw data retained')
    return revised
