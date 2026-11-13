"""Keep missingness, physical review and supervision eligibility separate."""
import numpy as np


def supervision_view(observed, names, needs_review):
    observed=np.asarray(observed,dtype=bool)
    review=np.zeros_like(observed)
    for j,name in enumerate(names):
        if name in {'ft','phasemargin'}:review[:,j]=needs_review
    return {'observed_mask':observed.copy(),'quality_review_mask':review,
            'engineering_eligible':observed&~review,
            'interpretation':'Eligibility under a conservative review policy, not proof of invalidity or certification of remaining labels'}


def measured_unity_crossings(frequency,gain_db,phase_rad=None,*,is_signed_loop_gain=False):
    """Never extrapolate. PM requires an explicitly identified loop-gain transfer.

    Interpolation is linear in dB versus log frequency. This definition is
    versioned and must not silently replace official legacy extraction labels.
    """
    f=np.asarray(frequency,dtype=float);g=np.asarray(gain_db,dtype=float)
    if f.ndim!=1 or f.shape!=g.shape or len(f)<2 or not np.isfinite(f).all() or not np.isfinite(g).all() or not (f>0).all() or not (np.diff(f)>0).all():raise ValueError('Invalid AC scan')
    phase=None
    if phase_rad is not None:
        phase=np.asarray(phase_rad,dtype=float)
        if phase.shape!=f.shape or not np.isfinite(phase).all():raise ValueError('Invalid phase scan')
        phase=np.unwrap(phase)
        phase-=2*np.pi*np.round(phase[0]/(2*np.pi))
    down=np.flatnonzero((g[:-1]>=0)&(g[1:]<0))
    rows=[]
    for i in down:
        weight=float(g[i]/(g[i]-g[i+1]));hz=float(np.exp(np.log(f[i])+weight*np.log(f[i+1]/f[i])))
        pm=None
        if is_signed_loop_gain and phase is not None:
            if abs(phase[0])>np.pi/4:raise ValueError('Loop-gain phase/sign convention requires explicit review')
            pm=float(180+np.degrees(phase[i]+weight*(phase[i+1]-phase[i])))
        rows.append({'frequency_Hz':hz,'phase_margin_degree':pm,'bracket_Hz':[float(f[i]),float(f[i+1])]})
    return {'status':'measured_crossing' if rows else 'no_measured_descending_crossing',
      'crossings':rows,'single_crossing':len(rows)==1,'phase_definition_verified':bool(is_signed_loop_gain and phase is not None),
      'extrapolation_used':False}
