"""Evidence-based graph annotations; no simulation responses or label features."""
import copy
import math
import re
from collections import Counter
from .common import ROOT
from .netlist import FEATURE_NAMES, evaluate

VERSION='lecture12_features_v1'
ROLES=['positive_supply','negative_supply','ground','input','output','bias']
MODELS=['balun','capacitor','inductor','isource','mutual_inductor','nmos','pmos','port','resistor','vsource']
EXTRA=['finger_count','width_length_ratio','hierarchy_depth']+['device_type:'+m for m in MODELS]
LOG10={'width_m','length_m','resistance_ohm','capacitance_F','inductance_H','width_length_ratio'}
LOG2={'multiplicity','finger_count'}


def vocabularies(graphs):
    relations=set();terminals=set()
    for g in graphs:
        for d in g['devices']:
            terminals.update(d['model']+':'+p for p in d['ports'])
        for e in g['edges']:
            model=g['devices'][e['device']]['model'];a,b=e['ports']
            relations.update([model+':'+a+'>'+b,model+':'+b+'>'+a])
    return {'relations':sorted(relations),'terminal_types':sorted(terminals),'device_types':MODELS}


def annotate_roles(graph,spec):
    annotations={n:{'roles':[],'evidence':[]} for n in graph['nets']}
    def mark(net,role,kind,detail):
        if net not in annotations:return
        a=annotations[net]
        if role not in a['roles']:a['roles'].append(role)
        record={'role':role,'basis':kind,'detail':detail}
        if record not in a['evidence']:a['evidence'].append(record)
    if spec['source_dataset']=='LOADBench':
        text=(ROOT/graph['source_netlist']).read_text(encoding='utf-8')
        line=next(s.strip() for s in text.splitlines() if s.strip().lower().startswith('xopamp '))
        fields=line.lower().split();formal=graph['external_ports']
        assert len(fields)==len(formal)+2
        bindings=dict(zip(formal,fields[1:-1]))
        assert bindings['sourcenmos']=='0' and bindings['sourcepmos']=='vdd!'
        assert re.search(r'(?im)^\s*vdd\s+vdd!\s+0\s+dc\s+\{vdd\}',text)
        assert spec['conditions']['supply_voltage_V']>0
        mark('sourcenmos','ground','testbench_binding',line+'; SPICE node 0')
        mark('sourcepmos','positive_supply','testbench_binding',line+'; Vdd referenced to node 0')
        for p in ['in1','in2']:mark(p,'input','DUT_port_and_testbench',p+' -> '+bindings[p])
        mark('out','output','DUT_port_and_testbench','out -> '+bindings['out']+'; output load and feedback in fixture')
        mark('ibias','bias','DUT_port_and_testbench','ibias -> '+bindings['ibias'])
    else:
        mark('0','ground','SPICE_reference','Global reference node 0')
        # A DC zero-voltage reference is evidence of role, not permission to
        # merge two nodes. Exclude AC-excited voltage sources.
        for _ in range(len(graph['devices'])+1):
            changed=False
            for d in graph['devices']:
                if d['model']!='vsource' or len(d['nets'])!=2:continue
                attrs=d['attributes']
                if 'mag' in attrs or attrs.get('type','dc').strip('"')!='dc':continue
                try:value=evaluate(attrs.get('dc','0'),spec.get('defaults',{}))
                except KeyError:continue
                if value!=0:continue
                a,b=d['nets']
                for ref,target in [(a,b),(b,a)]:
                    if 'ground' in annotations[ref]['roles'] and 'ground' not in annotations[target]['roles']:
                        mark(target,'ground','zero_DC_reference_source',d['name']+'; '+str(d['nets']))
                        changed=True
            if not changed:break
        for net in graph['nets']:
            leaf=net.rsplit(':',1)[-1].replace('\\','').lower().rstrip('!')
            if leaf in {'gnd','ground'} and 'ground' not in annotations[net]['roles']:
                mark(net,'ground','net_name_convention','Unverified ground name; not assumed shorted to 0')
            if leaf in {'vdd','vcc','vdda','vddd','avdd','dvdd'}:
                mark(net,'positive_supply','net_name_convention','Supply net name '+net)
            if leaf in {'vss','vee','avss','dvss'} and 'ground' not in annotations[net]['roles']:
                mark(net,'negative_supply','net_name_convention','Lower/negative rail name, not assumed zero voltage')
            if ':' not in net:
                if re.fullmatch(r'(?:vin|in|inp|inn)(?:[+\-]|\d+)?',leaf) or re.fullmatch(r'(?:rf|lo)(?:[+\-])?',leaf):
                    mark(net,'input','net_name_convention','Top-level signal name '+net)
                if re.fullmatch(r'(?:vout|out)(?:[+\-]|\d+)?',leaf) or leaf in {'if_out'}:
                    mark(net,'output','net_name_convention','Top-level signal name '+net)
            if re.fullmatch(r'(?:vbias\d*|vb\d+|vcont|ibias)',leaf):
                mark(net,'bias','net_name_convention','Bias/control name '+net)
        for d in graph['devices']:
            if d['model']!='vsource' or len(d['nets'])!=2:continue
            a,b=d['nets'];leaf=a.rsplit(':',1)[-1].lower().rstrip('!')
            if 'ground' not in annotations[b]['roles']:continue
            try:value=evaluate(d['attributes'].get('dc','0'),spec.get('defaults',{}))
            except KeyError:continue
            if value<=0:continue
            if leaf in {'vdd','vcc','vdda','vddd','avdd','dvdd'}:
                mark(a,'positive_supply','DC_supply_source',d['name']+' '+str(d['nets'])+' dc='+d['attributes'].get('dc','0'))
            elif 'mag' not in d['attributes'] and d['attributes'].get('type','dc')=='dc':
                # Unnamed drain rail (e.g. class-E PA): a constant DC source
                # feeds a MOS drain through an inductor/choke. Bias-only gate
                # sources and AC excitations are not promoted to supply rails.
                for choke in graph['devices']:
                    if choke['model']!='inductor' or a not in choke['nets']:continue
                    other=[n for n in choke['nets'] if n!=a]
                    if any(m['model'] in {'nmos','pmos'} and m['nets'][0] in other for m in graph['devices']):
                        mark(a,'positive_supply','DC_drain_feed_through_inductor',d['name']+' -> '+choke['name']+' -> MOS D; source referenced to ground')
    return annotations


def enrich_graph(graph,spec,vocab):
    g=copy.deepcopy(graph)
    # Idempotent: always build feature extensions from the original SI schema.
    g['feature_names']=list(FEATURE_NAMES)+EXTRA
    g['device_hierarchy_depth']=[]
    for i,d in enumerate(g['devices']):
        depth=d['name'].count('/')
        g['device_hierarchy_depth'].append(depth)
        attrs=d['attributes'];mos=d['model'] in {'nmos','pmos'}
        ratio=f"({attrs['w']})/({attrs['l']})" if mos else None
        extra=[attrs.get('nf') if mos else None,ratio,str(depth)]
        extra += ['1' if d['model']==m else '0' for m in MODELS]
        g['feature_expressions'][i]=g['feature_expressions'][i][:len(FEATURE_NAMES)]+extra
    roles=annotate_roles(g,spec)
    node_names=['connected_device_count','is_supply','is_positive_supply','is_negative_supply','is_ground',
                'is_input','is_output','is_bias','role_unassigned','has_verified_role_evidence',
                'role_evidence_convention_only']+['terminal_count:'+p for p in vocab['terminal_types']]
    g['node_feature_names']=node_names;g['node_features']=[];g['node_annotations']=[]
    for net in g['nets']:
        attached=[d for d in g['devices'] if net in d['nets']]
        counts=Counter(d['model']+':'+p for d in attached for p,n in zip(d['ports'],d['nets']) if n==net)
        a=roles[net];r=set(a['roles'])
        verified=any(e['basis']!='net_name_convention' for e in a['evidence'])
        flags=[int(bool(r&{'positive_supply','negative_supply','ground'}))]+[int(role in r) for role in ROLES]
        g['node_features'].append([len(attached)]+flags+[int(not r),int(verified),int(bool(r) and not verified)]+[counts[t] for t in vocab['terminal_types']])
        g['node_annotations'].append({'net':net,'node_type':'unassigned' if not r else '+'.join(sorted(r)),
            **a,'role_semantics':'Positive role flags; zero means unassigned, not proof of absence',
            'has_verified_evidence':verified})
    rel_index={r:i for i,r in enumerate(vocab['relations'])}
    mp={'edge_index':[[],[]],'edge_device_index':[],'edge_base_index':[],'edge_port_pairs':[],'edge_type':[],
        'reverse_edge_index':[],'edge_direction':[]}
    for i,e in enumerate(g['edges']):
        model=g['devices'][e['device']]['model'];a,b=e['ports'];j=len(mp['edge_type'])
        for u,v,p,q,direction in [(e['source'],e['target'],a,b,1),(e['target'],e['source'],b,a,-1)]:
            mp['edge_index'][0].append(u);mp['edge_index'][1].append(v)
            mp['edge_device_index'].append(e['device']);mp['edge_base_index'].append(i)
            mp['edge_port_pairs'].append([p,q]);mp['edge_type'].append(rel_index[model+':'+p+'>'+q])
            mp['edge_direction'].append(direction)
        mp['reverse_edge_index'] += [j+1,j]
    deg=Counter(mp['edge_index'][1])
    mp['symmetric_norm_weight']=[1/math.sqrt(deg[u]*deg[v]) for u,v in zip(*mp['edge_index'])]
    mp['normalization_note']='Optional structural weights, including retained typed self-loops. Not automatically applied by a model.'
    g.update(annotation_version=VERSION,feature_vocabulary=vocab,message_passing=mp,
        hierarchy_scope='DUT primitives; PDK wrappers treated as primitive devices' if spec['source_dataset']=='LOADBench' else 'flattened source netlist; top-level depth zero',
        num_message_edges=len(mp['edge_type']),
        message_edge_policy='Two typed orientations per physical port pair, including tied-port self-loops; not an inferred signal-flow direction',
        feature_transform_policy='Raw SI retained; log10 positive W/L/R/C/L/ratio, log2 positive m/nf; zero gets a separate flag; no fitted normalization')
    return g


def transform_features(features,names):
    values=[];known=[];zero=[]
    for row in features:
        v=[];k=[];z=[]
        for value,name in zip(row,names):
            iszero=value is not None and value==0 and name in LOG10|LOG2
            valid=value is not None and (name not in LOG10|LOG2 or value>=0)
            if not valid:converted=None
            elif iszero:converted=0.
            elif name in LOG10:converted=math.log10(value)
            elif name in LOG2:converted=math.log2(value)
            else:converted=value
            v.append(converted);k.append(valid);z.append(iszero)
        values.append(v);known.append(k);zero.append(z)
    return values,known,zero
