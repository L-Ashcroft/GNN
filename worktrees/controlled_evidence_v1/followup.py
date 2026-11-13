"""Observe P1/P2, score frozen tests, then serially run the four P3/P4 jobs.

Not named supervise/watch: the existing pinned scheduler must not mistake this
lightweight waiting observer for a competing trainer. Explicit checks prevent
starting a new trainer until its original five jobs and supervisor have exited.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,time,json,hashlib,subprocess,msvcrt
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
CONTROL=ROOT/'monitoring/controlled_evidence_v1'
from assessment import read,sha,atomic,evaluate_run,compare_intersection,ready,DEST
M04='experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4'
BASE=['rgcn_memory_safe_loto_v2','rgcn_memory_safe_id_control_v2','rgcn_memory_safe_no_film_v2','rgcn_memory_safe_static_edges_v2','rgcn_memory_safe_collapsed_relations_v2']
LABELS={'rgcn_memory_safe_loto_v2':'P1 M02 LOTO','rgcn_memory_safe_id_control_v2':'P1/P2 完整神经头ID',
        'rgcn_memory_safe_no_film_v2':'P2 No-FiLM','rgcn_memory_safe_static_edges_v2':'P2 Static-Edge',
        'rgcn_memory_safe_collapsed_relations_v2':'P2 Merged-Relation','controlled_single_rgcn_lb_v1':'P3/P4 LOADBench单源RGCN',
        'controlled_multi_rgcn_lb_v1':'P3 多源RGCN→LOADBench','controlled_single_mlp_lb_v1':'P4 参数MLP（LOADBench）',
        'controlled_single_gcn_lb_v1':'P4 GCN（LOADBench）'}
def now():return datetime.now(timezone.utc).isoformat()
def live(pid,created):
    try:return abs(psutil.Process(pid).create_time()-created)<.01
    except (psutil.NoSuchProcess,psutil.AccessDenied,TypeError):return False
def other_workers():
    result=[]
    for p in psutil.process_iter(['cmdline','name','create_time']):
        if p.pid in {os.getpid(),os.getppid()}:continue
        args=[str(a).replace('\\','/').lower() for a in p.info['cmdline'] or []]
        if any(a.startswith((ROOT/'worktrees').as_posix().lower()+'/') and a.endswith(('/train.py','/supervise.py','/supervisor.py','/watch.py')) for a in args):result.append(p.pid)
    return result
def comparisons():
    # Never let head routing differences be confused with model differences.
    full=DEST/'rgcn_memory_safe_id_control_v2/summary.json'
    if full.exists():
        reference=read(full)
        for name in BASE[2:]:
            p=DEST/name/'summary.json'
            if p.exists():
                other=read(p);assert reference['test_ids_sha256']==other['test_ids_sha256'] and reference['labels_mask_sha256']==other['labels_mask_sha256']
    a,b=[DEST/('controlled_'+kind+'_rgcn_lb_v1')/'summary.json' for kind in ('single','multi')]
    if a.exists() and b.exists():
        x,y=read(a),read(b);assert x['test_ids_sha256']==y['test_ids_sha256'] and x['labels_mask_sha256']==y['labels_mask_sha256']
        sa,sb=[read(ROOT/'artifacts'/('controlled_'+kind+'_rgcn_lb_v1')/'scope.json') for kind in ('single','multi')]
        assert sa['sources']['train']['LOADBench']==sb['sources']['train']['LOADBench']
        na,nb=[read(ROOT/'artifacts'/('controlled_'+kind+'_rgcn_lb_v1')/'normalization.json') for kind in ('single','multi')]
        import numpy as np
        for key in ('center','std','count'):assert np.allclose(na['inputs'][2][key],nb['inputs'][2][key],rtol=1e-10,atol=1e-10)
        # All LOADBench target calibration must match, despite extra source heads.
        tasks=read(ROOT/'artifacts/controlled_single_rgcn_lb_v1/task_registry.json')
        for g,t in tasks.items():
            if t['contract']['source_dataset']=='LOADBench':
                for key in ('center','std','count'):assert np.allclose(na['targets'][g][key],nb['targets'][g][key],rtol=1e-10,atol=1e-10)
        atomic(DEST/'source_comparison.json',{'same_test_ids_and_labels':True,'same_LOADBench_train_rows':sa['sources']['train']['LOADBench'],
             'extra_source_distinct_train_rows':sb['sources']['train']['FALCON']+sb['sources']['train']['AICircuit'],
             'single_selected_updates':x['selected_updates'],'multi_selected_updates':y['selected_updates'],
             'single_actual_completed_updates':x['training_steps'],'multi_actual_completed_updates':y['training_steps'],
             'same_maximum_steps':31260,'hit5_delta_multi_minus_single_pp':100*(y['task_macro']['macro_hit5']-x['task_macro']['macro_hit5']),
             'remaining_confounder':'Single cycles LOADBench more often; same unique target-label pool and per-epoch steps, not same target presentations. Early stopping/selected epoch can differ.'})
    compare_intersection('rgcn_memory_safe_id_control_v2','rgcn_memory_safe_loto_v2')
    compare_intersection('rgcn_memory_safe_id_control_v2',M04)
def publish(names,state):
    pre=read(ROOT/'artifacts/controlled_evidence_v1/preflight.json')
    counts={j['id']:j['model_parameters'] for j in pre['jobs']}
    counts.update(dict(zip(BASE,[8146624,8146624,8014528,7883456,1980224])))
    lines=['# 受控实验结果与证据状态','',f'更新：{datetime.now().astimezone(timezone.utc).isoformat()}（UTC）。本文件由独立续行队列根据实际产物更新；“待测”不是0分。',
      '', 'P1/P2继续原串行队列及原recipe；P3/P4只有在前五项完成、旧监督器退出且资源满足后才启动。当前阶段：'+state+'.',
      '', '| 实验 | 状态 | 参数量 | 最优epoch | 测试hit@5%／10% | R²均值／中位数／负值项 | 完整MRE／中位／P95（%） | 实际训练步／训练小时 |',
      '| --- | --- | ---: | ---: | --- | --- | --- | --- |']
    def fmt(v):return '—' if v is None else f'{v:.4g}'
    for name in [M04]+names:
        p=DEST/name/'summary.json';s=read(ROOT/'artifacts'/name/'status.json') if (ROOT/'artifacts'/name/'status.json').exists() else {}
        label='既有M04首折（独立列示）' if name==M04 else LABELS[name]
        if p.exists():
            r=read(p);m=r['task_macro'];d=r['sample_equal_native_MRE']['untrimmed_sample_mre']
            lines.append(f"| {label} | 冻结测试已核验 | {r['parameters'] or counts.get(name,'—')} | {r['selected_epoch']} | {100*m['macro_hit5']:.2f}%／{100*m['macro_hit10']:.2f}% | {fmt(m['macro_r2'])}／{fmt(m['median_r2'])}／{m['negative_r2_tasks']} | {fmt(d['mean_pct'])}／{fmt(d['median_pct'])}／{fmt(d['p95_pct'])} | {r['training_steps']}／{r['training_step_seconds']/3600:.2f} |")
        else:lines.append(f"| {label} | {s.get('state','等待前序任务')}；epoch {s.get('epoch','—')} batch {s.get('batch','—')} | {counts.get(name,'—')} | — | 待测 | 待测 | 待测 | — |")
    lines += ['', '训练小时仅累计已完成轮的训练Step，未包括验证、资源等待及中断轮；最终成本以history和resource_summary为准。MAE/RMSE按指标原单位保存在各evaluation目录的per_task.parquet及native_per_topology_task.parquet，不跨单位汇总。',
       '', '## 对照范围与公平性',
       '', '- P2保持既有公开ID分区、129组路由、神经头、损失、种子42、最多20轮、最少4轮及耐心5不变。三项消融各从头训练。No-FiLM仍保留域嵌入；Merged-Relation仍含节点端口统计且参数量较小，它不是标准GCN。',
       '- P3单源与多源使用完全相同的160,046条LOADBench训练设计、20,653条验证、19,301条测试。多源另加160,051条FALCON及79,986条AICircuit训练记录，全部保持独立指标语义。验证选模仅看LOADBench。',
       '- P3每轮均400,083次训练呈现／1,563次更新，最多31,260次；单源循环原TRAIN池，未创建重复数据。两组独特LOADBench标签量相同，但单源每轮看见目标记录更频繁；早停和选中epoch亦可能不同。报告实际更新数，不声称严格等目标呈现次数。',
       '- P3/P4新版本共用8维已知条件（温度、供电、负载、偏流的固定尺度变换与known标记）及线性条件分支；未知值不推断。RGCN比P2多1,152参数，故P4参照是该新版本单源RGCN，不能直接把不同版本的分数相减。',
       '- P4先限定LOADBench已知拓扑ID任务，MLP按训练池最大器件数零填充，以固定器件顺序、全部90维器件输入和present标记构成参数向量；无邻接或消息传递。不能用该结果证明跨拓扑泛化。',
       '- GCN核心为D^-1/2(A+I)D^-1/2HW；保留原多重边、自环并加单位自环，器件边特征在输入注入节点并进入边池化。残差、LayerNorm、FiLM及头框架共用，不是完全不含边参数的GCN，也不是端口关系合并版。',
       '- P1 ID—LOTO只比较sample_id共同测试交集，并核对标签、Mask及监督语义；全局ID和单折LOTO总体不直接相减。训练规模、验证选择范围、旧M04微批/实现差异仍需列示。',
       '', '## 六个研究问题的当前答案', '']
    for q,reference,candidates in [
        ('端口关系区分','rgcn_memory_safe_id_control_v2',['rgcn_memory_safe_collapsed_relations_v2']),
        ('动态边更新','rgcn_memory_safe_id_control_v2',['rgcn_memory_safe_static_edges_v2']),
        ('FiLM','rgcn_memory_safe_id_control_v2',['rgcn_memory_safe_no_film_v2']),
        ('多源训练','controlled_multi_rgcn_lb_v1',['controlled_single_rgcn_lb_v1']),
        ('完整RGCN与简单模型','controlled_single_rgcn_lb_v1',['controlled_single_mlp_lb_v1','controlled_single_gcn_lb_v1'])]:
        rp=DEST/reference/'summary.json';entries=[]
        for candidate in candidates:
            cp=DEST/candidate/'summary.json'
            if not (rp.exists() and cp.exists()):continue
            r,c=read(rp),read(cp);assert r['test_ids_sha256']==c['test_ids_sha256'] and r['labels_mask_sha256']==c['labels_mask_sha256']
            delta=100*(r['task_macro']['macro_hit5']-c['task_macro']['macro_hit5'])
            rm,cm=[v['sample_equal_native_MRE']['untrimmed_sample_mre']['mean_pct'] for v in (r,c)]
            entries.append(f"相对{LABELS[candidate]}，参考模型hit@5%差{delta:+.3f}个百分点；完整MRE {rm:.6g}%／{cm:.6g}%，负R²项{r['task_macro']['negative_r2_tasks']}／{c['task_macro']['negative_r2_tasks']}。负命中率差表示该口径下参考模型更差。")
        lines.append('- '+q+'：'+(' '.join(entries)+' 本比较限于单种子、当前容量与已声明预算。' if entries else '尚无完整冻结配对，不能判断收益。'))
    lines.append('- 未知拓扑相对可比ID：只使用下列共同测试交集，不引用历史局部读出的54.32%作为神经ID参照。')
    if (DEST/'source_comparison.json').exists():lines+=['','来源配对结果：`artifacts/controlled_evidence_v1/evaluation/source_comparison.json`（负值增益表示负迁移）。']
    for p in sorted((DEST/'matched_ID_LOTO').glob('*.json')) if (DEST/'matched_ID_LOTO').exists() else []:
        r=read(p);lines+=['',f"ID—LOTO共同测试：{p.name}，{r['common_test_samples']}条；ID hit@5%={r['ID']['task_macro']['macro_hit5']*100:.2f}%，LOTO={r['LOTO']['task_macro']['macro_hit5']*100:.2f}%。仍受训练群体和选模范围差异影响。"]
    lines += ['', '## 归档与尚缺证据','',
       '每个新运行保留recipe、scope、训练统计、model_summary、history、last/best权重、selection_frozen及测试预测。额外评价结果只在artifacts/controlled_evidence_v1/evaluation/下写入，检查选中权重/recipe哈希及原始产物未变。中间验证不会填入测试列。',
       '', '仍需弱化的结论：尚不能仅凭历史局部读出证明端口RGCN内部组件有效，不能仅凭数据覆盖证明多源提高精度，也不能用两折代替全面未知拓扑评价。P1—P4未完成前不新增上述性能贡献。当前不扩展Few-shot重复或新增复杂模型，不改论文正文。',
       '', '实时状态：monitoring/controlled_evidence_v1/status.json；原队列：monitoring/public_evidence_queue_v1/status.json；小批正确性：artifacts/controlled_evidence_v1/preflight.json。小批检查不生成正式权重，不使用验证/测试标签。','']
    (ROOT/'reports/controlled_experiment_results.md').write_text('\n'.join(lines),encoding='utf-8')
    progress=ROOT/'reports/public_experiment_completion_status.md';text=progress.read_text(encoding='utf-8')
    marker='<!-- CONTROLLED_EVIDENCE_AUTO -->'
    block=marker+'\n\n## P1—P4受控实验自动更新\n\n更新：'+now()+'。阶段：'+state+'；原队列保持优先，未来四项已实现并通过真实TRAIN小批检查。当前完整对照结果见[受控实验结果](controlled_experiment_results.md)，实时状态见[续行队列](../monitoring/controlled_evidence_v1/status.json)。\n\n'+marker
    if marker in text:
        a=text.index(marker);b=text.index(marker,a+len(marker))+len(marker);text=text[:a]+block+text[b:]
    else:text+='\n'+block+'\n'
    progress.write_text(text,encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    suite=read(HERE/'suite.json');names=BASE+[j['id'] for j in suite['jobs']]
    pins=read(ROOT/'artifacts/controlled_evidence_v1/preflight.json')['source_sha256']
    assert read(ROOT/'artifacts/controlled_evidence_v1/preflight.json')['passed']
    state={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'state':'observing_P1_P2','training_active':False}
    policy={'pins':pins,'serial_only':True,'wait_for_original_queue_completion':True,'CPU_only':True,'no_new_FewShot_or_large_LOTO':True}
    if (CONTROL/'policy.json').exists():assert read(CONTROL/'policy.json')==policy,'Pinned follow-up policy changed'
    else:atomic(CONTROL/'policy.json',policy)
    child=None;current=None;failures=0;stable=None
    try:
        while True:
            assert all(sha(ROOT/p)==h for p,h in pins.items()),'Future source/config changed; no cross-recipe continuation'
            if (CONTROL/'STOP').exists():
                if child and child.poll() is None:(ROOT/current['output']/'STOP').write_text('User stop; save at batch boundary',encoding='utf-8');time.sleep(5);continue
                state['state']='stopped';break
            if psutil.virtual_memory().available>=1.5*2**30:
                for name in [M04]+names:evaluate_run(name)
                comparisons()
            if child is not None and child.poll() is not None:
                f=ROOT/current['output'];s=read(f/'status.json') if (f/'status.json').exists() else {}
                if s.get('state')!='complete':
                    error=(s.get('error_type','')+' '+s.get('error','')).lower()
                    if child.returncode!=75 and s.get('state')!='paused' and not any(w in error for w in ['memoryerror','paging file','out of memory']):raise RuntimeError('Non-resource training failure: '+current['id']+' '+str(s))
                    failures+=1;state['cooldown_until']=time.time()+120
                child=None;stable=None
            old=read(ROOT/'monitoring/public_evidence_queue_v1/status.json')
            old_complete=all((ROOT/'artifacts'/n/'status.json').exists() and read(ROOT/'artifacts'/n/'status.json').get('state')=='complete' for n in BASE)
            old_alive=live(old.get('supervisor_pid'),old.get('supervisor_create_time'))
            if child and child.poll() is None:
                status_path=ROOT/current['output']/'status.json';s=read(status_path) if status_path.exists() else {}
                state.update(state='training_future_job',current_job=current['id'],worker=s,training_active=s.get('state')=='training' and live(s.get('pid'),s.get('create_time')))
            elif not old_complete or old_alive:
                state.update(state='observing_P1_P2',original_job=old.get('current_job'),original_state=old.get('worker_status',{}),training_active=False)
            else:
                pending=[j for j in suite['jobs'] if not (ROOT/j['output']/'status.json').exists() or read(ROOT/j['output']/'status.json').get('state')!='complete']
                if not pending:
                    fully_scored=all((DEST/name/'receipt.json').exists() for name in [M04]+names)
                    state.update(state='suite_complete' if fully_scored else 'training_complete_waiting_scoring_resources',training_active=False)
                    publish(names,state['state'])
                    if fully_scored:break
                    state['updated_at']=now();atomic(CONTROL/'status.json',state)
                    if args.once:break
                    time.sleep(30);continue
                current=pending[0];gate=2.2 if failures>=4 else 1.8 if failures>=2 else 1.4
                others=other_workers();mem=psutil.virtual_memory().available/2**30
                okay=not others and mem>=gate and time.time()>=state.get('cooldown_until',0)
                stable=(stable or time.monotonic()) if okay else None
                state.update(state='waiting_resources_or_existing_worker',current_job=current['id'],training_active=False,other_workers=others,available_RAM_GiB=mem,required_RAM_GiB=gate)
                if stable and time.monotonic()-stable>=20 and not args.once:
                    folder=ROOT/current['output'];folder.mkdir(parents=True,exist_ok=True)
                    command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'train.py'),'--config',str(ROOT/current['config']),'--backend','cpu']
                    if (folder/'recipe.json').exists():command.append('--resume')
                    with (CONTROL/(current['id']+'_stdout.log')).open('ab') as out,(CONTROL/(current['id']+'_stderr.log')).open('ab') as err:
                        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                    with (CONTROL/'events.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps({'at':now(),'event':'launch','job':current['id'],'command':command,'launcher_pid':child.pid})+'\n')
            state['updated_at']=now();atomic(CONTROL/'status.json',state);publish(names,state['state'])
            if args.once:break
            time.sleep(30)
    except BaseException as e:
        if child and child.poll() is None:(ROOT/current['output']/'STOP').write_text('Control error; retain checkpoint for review',encoding='utf-8')
        state.update(state='requires_review',training_active=False,error_type=type(e).__name__,error=str(e));atomic(CONTROL/'status.json',state);raise
    finally:
        state['updated_at']=now();atomic(CONTROL/'status.json',state)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
