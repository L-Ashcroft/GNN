"""Mirror the real continuation state; keep old tested-model results distinct."""
from pathlib import Path
import json
import time
import os
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
RUN='experiment1_cached_head_cpu_continue_v1';OUT=ROOT/'artifacts'/RUN
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    temp=path.with_suffix(path.suffix+'.view.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)
def main():
    target=ROOT/'monitoring/current_training.json';previous=read(target)
    tested_model='artifacts/experiment1_accuracy_v2_cached_head/best.pt'
    while True:
        current=read(target)
        if current.get('run_id') not in {None,RUN}:return
        try:
            status=read(OUT/'status.json');supervisor=read(ROOT/'monitoring'/RUN/'status.json')
            phase=status['state'];complete=phase=='complete'
            pointer={'run_id':RUN,'state':phase,'updated_at':datetime.now(timezone.utc).isoformat(),
                'runtime_status':str((OUT/'status.json').relative_to(ROOT)).replace('\\','/'),
                'monitor_status':f'monitoring/{RUN}/status.json','device':'cpu','body_frozen':True,
                'head_epoch_total':status.get('epoch'),'additional_epoch':status.get('additional_epoch'),
                'batch':status.get('batch'),'batches_per_epoch':status.get('batches_per_epoch'),
                'model':tested_model,'model_evaluation_status':'Previously tested frozen model remains preserved',
                'candidate_model':str((OUT/'best.pt').relative_to(ROOT)).replace('\\','/'),
                'candidate_validation_hit5':status.get('best_validation_score'),'candidate_best_head_epoch':status.get('best_total_head_epoch'),
                'candidate_test_metrics':None,'previous_test_metrics':'artifacts/experiment1_accuracy_v2_cached_head/test_metrics.json',
                'test_policy':'Continuation uses train/validation only; previous test score does not describe changed candidate weights',
                'implementation':'versions/rgcn_calibrated_head_v1','results_report':'reports/cached_head_continuation.md'}
            if complete and status.get('best_total_head_epoch',6)>6:
                pointer['model']=pointer['candidate_model'];pointer['model_evaluation_status']='Selected by validation only; no new test score'
            write(target,pointer)
            if complete or supervisor['state'] in {'worker_error_requires_review','restart_limit_reached','user_stop_requested','source_changed_requires_review'}:return
        except (FileNotFoundError,PermissionError,json.JSONDecodeError):pass
        time.sleep(15)

if __name__=='__main__':main()
