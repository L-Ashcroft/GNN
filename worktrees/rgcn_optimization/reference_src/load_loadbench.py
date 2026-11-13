"""LOADBench real CSV and port-preserving DUT adapter; no synthetic labels."""
import re
import numpy as np
from .common import digest
from .netlist import evaluate,make_graph,feature_expressions,FEATURE_NAMES,attributes

METRICS=[('dc_gain','dB'),('ft','Hz'),('p','W'),('phasemargin','degree'),
         ('sr_neg','V/s'),('sr_pos','V/s'),('voffset','V'),('icmr_min','V'),
         ('icmr_max','V'),('vout_min','V'),('vout_max','V'),('area','um2_estimated'),
         ('cmrr','dB'),('psrr','dB')]


def quality_mask(frame):
    present=[k for k,u in METRICS if k in frame]
    if not present:return np.zeros(len(frame),dtype=bool)
    values=frame[present].to_numpy(dtype=float)
    # Extractor uses +/-1e13 when measurements fail. A failed measurement is
    # quarantined, never misrepresented as an absent metric via Mask=False.
    ok=np.isfinite(values).all(axis=1)&(np.abs(values)<1e12).all(axis=1)
    for col in [c for c in frame if c.lower().startswith('param_')]:
        x=frame[col].to_numpy(dtype=float)
        ok&=np.isfinite(x)&((x>=0) if col.lower().startswith('param_c') else (x>0))
    for c in ('ft','p','area'):
        if c in frame:ok&=frame[c].to_numpy()>0
    if 'sr_pos'in frame:ok&=frame.sr_pos.to_numpy()>=0
    if 'sr_neg'in frame:ok&=frame.sr_neg.to_numpy()<=0
    if {'icmr_min','icmr_max'}<=set(frame):ok&=frame.icmr_min.to_numpy()<=frame.icmr_max.to_numpy()
    if {'vout_min','vout_max'}<=set(frame):ok&=frame.vout_min.to_numpy()<=frame.vout_max.to_numpy()
    return ok


def definitions(text):
    result={}
    for raw in text.splitlines():
        line=raw.strip()
        if not line.lower().startswith('.param '):continue
        for k,v in attributes(line[7:]).items():result[k.lower()]=v.strip("'\"{}").lower()
    return result


def resolve(defs,overrides,allow_unresolved=False):
    values=dict(overrides);pending=dict(defs)
    while pending:
        progress=False
        for k,expression in list(pending.items()):
            try:v=values[k] if k in values else evaluate(expression,values)
            except KeyError:continue
            if not np.isfinite(v) or (v<0 if k.startswith('param_c') else v<=0):
                raise ValueError('Invalid LOADBench parameter '+k)
            values[k]=float(v);del pending[k];progress=True
        if not progress:
            if allow_unresolved:break
            raise ValueError('Unresolved LOADBench aliases '+str(pending))
    return dict(sorted(values.items()))


def csv_to_si(name,value):
    # Artifact unit audit: W/L CSV values are in micrometres; params.cir uses
    # SI literals (e.g. 28e-6); capacitances already use farads. Keep the original
    # CSV and parameter template so this conversion is independently reviewable.
    return float(value)*1e-6 if name.lower().startswith(('param_w','param_l')) else float(value)


def graph_from_combined(text):
    match=re.search(r'(?ims)^\s*\.subckt\s+(\S+)\s+([^\n]+)\n(.*?)^\s*\.ends\b[^\n]*',text)
    if not match:raise ValueError('DUT subckt missing')
    name=match[1];formal=match[2].lower().split();body=match[3]
    devices=[]
    for raw in body.splitlines():
        line=raw.strip().lower()
        if not line or line.startswith('*'):continue
        f=line.split();model=f[0][0]
        if model in {'x','m'}:
            if len(f)<6:raise ValueError('Truncated MOS instance')
            pdk=f[5]
            if 'nfet' not in pdk and 'pfet' not in pdk:raise ValueError('Unsupported subcircuit/model '+pdk)
            a=attributes(' '.join(f[6:]));core={k:v.strip("'\"{}") for k,v in a.items() if k in {'w','l','m','nf'}}
            if not {'w','l'}<=set(core):raise ValueError('Missing W/L')
            devices.append({'name':f[0],'nets':f[1:5],'ports':['D','G','S','B'],
                'model':'nmos' if 'nfet'in pdk else 'pmos','pdk_model':pdk,
                'attributes':core,'source_advanced_attributes':{k:v for k,v in a.items() if k not in core},'device_references':{}})
        elif model in {'r','c','l'}:
            if len(f)<4:raise ValueError('Truncated passive')
            value=' '.join(f[3:]).strip("'\"{}")
            devices.append({'name':f[0],'nets':f[1:3],'ports':['P','N'],
                'model':{'r':'resistor','c':'capacitor','l':'inductor'}[model],
                'attributes':{model:value},'device_references':{}})
        else:raise ValueError('Unsupported DUT line: '+raw)
    if not devices or len({d['name'] for d in devices})!=len(devices):raise ValueError('Invalid DUT devices')
    graph=make_graph(devices)
    graph.update(feature_names=FEATURE_NAMES,feature_expressions=feature_expressions(graph),
                 subcircuit_name=name,external_ports=formal,graph_scope='DUT; measurement fixtures kept as conditions')
    # Remove only the actual DUT text/name from the fixture fingerprint. Other
    # condition/extractor differences remain distinct semantic groups.
    fixture=(text[:match.start()]+text[match.end():]).replace(name,'DUT')
    fixture='\n'.join(s.strip().lower() for s in fixture.splitlines() if s.strip() and not s.lstrip().startswith('*'))
    return graph,digest(fixture)
