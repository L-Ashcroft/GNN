"""Streaming physical-unit metrics, separated by supervision contract and target."""
import math
import torch
from .losses import inverse_symlog

LOG_UNITS={'dB','dBm','dBc/Hz','source_label_dBm_compression_function'}


class MetricAccumulator:
    def __init__(self,tasks):self.tasks=tasks;self.values={}

    @torch.no_grad()
    def update(self,prediction,batch):
        physical=inverse_symlog(prediction.detach().cpu())*batch.scales.detach().cpu().to(torch.float64)
        target=batch.targets.detach().cpu().to(torch.float64);mask=batch.mask.detach().cpu()
        for i,group in enumerate(batch.group_ids):
            for slot in torch.nonzero(mask[i],as_tuple=False).flatten().tolist():
                rule=self.tasks[group]['slots'][str(slot)];p=float(physical[i,slot]);t=float(target[i,slot])
                if not math.isfinite(p) or not math.isfinite(t):raise ValueError('Nonfinite observed physical prediction')
                key=(group,rule['semantic_id'])
                a=self.values.setdefault(key,{'n':0,'mean':0.,'m2':0.,'sse':0.,'sae':0.,'relative_sum':0.,'relative_n':0,'hit5':0,'hit10':0,
                    'near_zero':0,'unit':rule['unit'],'original_name':rule['original_name'],'absolute_hit1':0})
                a['n']+=1;delta=t-a['mean'];a['mean']+=delta/a['n'];a['m2']+=delta*(t-a['mean'])
                error=abs(p-t);a['sse']+=error**2;a['sae']+=error
                a['absolute_hit1']+=int(error<=1.)
                # Relative errors of dB/logarithmic quantities are not physical
                # gain/power tolerances. Report MAE and absolute-unit hits instead.
                floor=1e-8*rule['scale']
                if abs(t)<=floor:a['near_zero']+=1
                elif rule['unit'] not in LOG_UNITS:
                    relative=error/abs(t);a['relative_n']+=1;a['relative_sum']+=relative
                    a['hit5']+=int(relative<=.05);a['hit10']+=int(relative<=.1)

    def compute(self):
        rows=[]
        for (group,semantic),a in sorted(self.values.items()):
            relative_n=a['relative_n']
            rows.append({'supervision_group':group,'semantic_id':semantic,'unit':a['unit'],'original_name':a['original_name'],
                'count':a['n'],'mae':a['sae']/a['n'],
                'r2':1.-a['sse']/a['m2'] if a['n']>=2 and a['m2']>0 else None,
                'mre':a['relative_sum']/relative_n if relative_n else None,
                'hit5':a['hit5']/relative_n if relative_n else None,'hit10':a['hit10']/relative_n if relative_n else None,
                'relative_count':relative_n,'near_zero_count':a['near_zero'],
                'hit_abs_1_unit':a['absolute_hit1']/a['n'] if a['unit'] in LOG_UNITS else None,
                'relative_policy':'not applicable to log units' if a['unit'] in LOG_UNITS else 'exclude near-zero targets with fixed engineering floor'})
        def mean(name):
            values=[r[name] for r in rows if r[name] is not None]
            return sum(values)/len(values) if values else None
        # Do not macro-average MAE across units, or pool heterogeneous raw values into R2.
        return {'per_target':rows,'macro_hit5':mean('hit5'),'macro_hit10':mean('hit10'),
                'macro_r2':mean('r2'),'macro_mre':mean('mre'),'target_groups':len(rows)}
