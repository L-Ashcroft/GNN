"""Native Windows low-memory scheduler; progressive actual model training."""
from pathlib import Path
import argparse
import ctypes
import msvcrt
import os
import subprocess
import time
import psutil
from common import ROOT, HERE, OUTPUT, NEW_CACHE, CONTROL, PARENT_OUTPUT, read, atomic, sha, pins, now


def identity(s):
    try:
        p=psutil.Process(s['pid'])
        return abs(p.create_time()-s['create_time'])<.001 and any(str(a).replace('\\','/').endswith('personal_refinement_v2/run.py') for a in p.cmdline())
    except (KeyError,psutil.NoSuchProcess,psutil.AccessDenied):return False


def snapshot(jobs):
    entries=[]
    for job in jobs:
        p=OUTPUT/job['id']/'status.json';s=read(p) if p.exists() else {'state':'queued'}
        entries.append({'job':job,'status':s})
    atomic(OUTPUT/'summary.json',{'updated_at':now(),'jobs':entries,'completed':sum(r['status']['state']=='complete' for r in entries),
           'total':len(entries),'test_used':False,'scope':'Known-topology GNN readout refinement and explicit physical-input controls'})
    lines=['# 个人模型修订与继续训练', '', '自动更新：'+now(), '',
           '本轮只用训练及验证数据。原模型、原测试结果、标签、Mask和划分不改写；不将本轮验证成绩当作固定测试成绩。',
           '低内存诊断：参数直达支路的验证hit@5%为50.72%、R²中位数0.831；这不是GNN结构增益。',
           '主要调整：按拓扑拟合训练统计，按拓扑平衡监督；对电压偏差、相位和dB量采用线性目标变换，其他量保留symlog；推理限制在训练标签支持范围。',
           '神经模型：原边中心RGCN保持独立检查点，冻结后提取图表示，与器件参数和已知条件共同输入线性直接项及残差MLP。',
           '本结构针对已知拓扑及有标签适配；不能用于LOTO，不能作为零样本模型。',
           '少样本每拓扑k条，共10k条；另用2,000条验证标签。各预算仅使用对应预算训练过的父模型，不从全量目标模型向少样本迁移。', '',
           '|任务|特征|状态|已完成轮次|选中轮次|验证hit@5%|选中R²中位数|',
           '|---|---|---|---:|---:|---:|---:|']
    for row in entries:
        j,s=row['job'],row['status'];hit=s.get('best_validation_hit5');r=s.get('selected_validation_median_r2')
        lines.append('|{}|{}|{}|{}|{}|{}|{}|'.format(j['id'],j['mode'],s['state'],s.get('completed_epochs','—'),s.get('best_epoch','—'),
            '—' if hit is None else f'{100*hit:.2f}%', '—' if r is None else f'{r:.3f}'))
    lines+=['','相同预算下的physical对照用于判断图表示的额外贡献；不得将全部提升归因于GNN。',
            '参数顺序固定到拓扑，因此本轮不是新的结构泛化证据；仍只有seed42。历史测试集已被此前模型评价，本程序不读取它进行调参。',
            '监控：`monitoring/personal_refinement_v2/status.json`；模型、recipe、训练历史及逐指标原单位误差见`artifacts/personal_refinement_v2/`。']
    (ROOT/'reports/personal_refinement_v2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    CONTROL.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'watcher.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert read(HERE/'checks.json')['passed']
    jobs=read(HERE/'suite.json')['jobs'];frozen=pins()
    policy={'source_sha256':frozen,'suite_sha256':sha(HERE/'suite.json'),'checks_sha256':sha(HERE/'checks.json'),
            'maximum_workers':1,'neural_readout_start_RAM_MiB':512,'GNN_extraction_start_RAM_GiB':1.6,
            'GPU_used':False,'test_access':False,'models_predeclared_before_test_evaluation':True}
    if (CONTROL/'policy.json').exists():assert read(CONTROL/'policy.json')==policy,'Source changed; refuse automatic cross-version restart'
    else:atomic(CONTROL/'policy.json',policy)
    state={'state':'starting','pid':os.getpid(),'create_time':psutil.Process().create_time(),'test_used':False}
    def update(**v):
        state.update(v,updated_at=now(),available_RAM_GiB=psutil.virtual_memory().available/2**30);atomic(CONTROL/'status.json',state)
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));child=None;active=None;retries={}
    try:
        while True:
            if (CONTROL/'STOP').exists() or sha(HERE/'suite.json')!=policy['suite_sha256'] or any(sha(ROOT/p)!=v for p,v in frozen.items()):
                if active:(OUTPUT/active['id']/'STOP').write_text('Explicit stop or source changed',encoding='utf-8')
                update(state='stopped');return
            snapshot(jobs)
            if child is not None:
                s=read(OUTPUT/active['id']/'status.json') if (OUTPUT/active['id']/'status.json').exists() else {}
                if child.poll() is None or identity(s):
                    update(state='training_active',current_job=active['id'],worker_identity_matches=identity(s),training=s)
                    atomic(ROOT/'monitoring/current_training.json',{'run_id':'personal_refinement_v2','state':s.get('state','starting'),
                           'updated_at':now(),'runtime_status':f'artifacts/personal_refinement_v2/{active["id"]}/status.json',
                           'monitor_status':'monitoring/personal_refinement_v2/status.json','test_used':False})
                    time.sleep(5);continue
                code=child.returncode
                if code!=0:
                    if code==75 or s.get('state')=='paused':
                        retries[active['id']]=retries.get(active['id'],0)+1
                        if retries[active['id']]>3:update(state='resource_retry_limit',training=s);return
                    else:update(state='worker_error_requires_review',exit_code=code,training=s);return
                child=None;active=None
            pending=[j for j in jobs if not (OUTPUT/j['id']/'status.json').exists() or read(OUTPUT/j['id']/'status.json')['state']!='complete']
            if not pending:update(state='suite_complete',completed_jobs=len(jobs));return
            available=psutil.virtual_memory().available
            # Work on low-memory parameter controls while frozen-GNN extraction waits.
            ready=[j for j in pending if j['mode']=='physical' or (NEW_CACHE/j['parent']/'manifest.json').exists()]
            if ready and available>=512*2**20:
                active=ready[0];folder=OUTPUT/active['id'];folder.mkdir(parents=True,exist_ok=True)
                if (folder/'STOP').exists():update(state='explicit_job_stop_preserved',current_job=active['id']);return
                command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'run.py'),'--job',active['id']]
                if (folder/'recipe.json').exists():command.append('--resume')
                with (folder/'stdout.log').open('ab') as out,(folder/'stderr.log').open('ab') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                update(state='worker_launched',current_job=active['id'],launcher_pid=child.pid);time.sleep(5);continue
            needed=next((j for j in pending if j['mode']!='physical' and not (NEW_CACHE/j['parent']/'manifest.json').exists()),None)
            if needed and available>=1.6*2**30:
                update(state='extracting_GNN_features',parent=needed['parent'])
                command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'watch.py'),
                         '--extract',needed['parent'],'--budget',str(needed['budget'])]
                with (CONTROL/'extract_stdout.log').open('ab') as out,(CONTROL/'extract_stderr.log').open('ab') as err:
                    extractor=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                while extractor.poll() is None:
                    update(state='extracting_GNN_features',parent=needed['parent'],extractor_pid=extractor.pid);time.sleep(5)
                if extractor.returncode not in [0,75]:update(state='embedding_error_requires_review',parent=needed['parent'],exit_code=extractor.returncode);return
                continue
            update(state='waiting_memory',pending_jobs=len(pending),required_RAM_GiB=1.6 if needed and not ready else .5);time.sleep(10)
    except BaseException as exc:update(state='watcher_error',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract');parser.add_argument('--budget');args=parser.parse_args()
    if args.extract:
        from embedding import prepare
        budget='full' if args.budget=='full' else int(args.budget)
        raise SystemExit(0 if prepare(args.extract,budget) else 75)
    main()
