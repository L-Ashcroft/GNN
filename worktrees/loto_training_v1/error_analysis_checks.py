"""Analytic RMSE reconstruction checks, no simulation or heldout data access."""
import math,json
from pathlib import Path
from error_analysis import reconstruct
tasks={'g':{'slots':{'0':{'semantic_id':'s'}}}}
train={'g::slot0':{'count':2,'mean':2.,'m2':8.}}
val={'g::slot0':{'count':2,'mean':2.,'m2':2.}}
base={'supervision_group':'g','semantic_id':'s','original_name':'gain','unit':'V','count':2,'mae':1.}
one=reconstruct([dict(base,r2=0.)],tasks,train,val)[0]
assert one['rmse']==1. and one['nrmse_train_std']==.5
two=reconstruct([dict(base,r2=-1.)],tasks,train,val)[0]
assert abs(two['rmse']-math.sqrt(2))<1e-14
undefined=reconstruct([dict(base,r2=None)],tasks,train,{'g::slot0':{'count':2,'mean':2.,'m2':0.}})[0]
assert undefined['rmse'] is None
rounded=reconstruct([dict(base,r2=1.)],tasks,train,val)[0]
assert rounded['rmse'] is None
result={'R2_to_RMSE_matches_analytic_predictions':True,'train_only_std_NRMSE_correct':True,'undefined_R2_not_fabricated':True,
    'rounded_R2_one_with_positive_MAE_not_reported_as_zero_RMSE':True,'test_data_used':False}
(Path(__file__).parent/'error_analysis_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
