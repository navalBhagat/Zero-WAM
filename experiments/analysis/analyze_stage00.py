"""Stage-00 cache compaction and memory plateau report; partial runs included."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.analysis.analyze import write_csv,log_records


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path)
    p.add_argument('--runs-root',type=Path);a=p.parse_args()
    data=json.loads(a.manifest.read_text());output=a.manifest.parent/'analysis';output.mkdir(exist_ok=True)
    summaries=[];events=[];hashes={};cache_rows=[]
    for run in data['runs']:
        root=a.runs_root or Path(run['environment']['EXPERIMENT_OUTPUT_ROOT'])
        folder=root/f'{run["task"]}_{run.get("job_id")}'
        config=json.loads((folder/'config.json').read_text()) if (folder/'config.json').exists() else {}
        timing=json.loads((folder/'job_timing.json').read_text()) if (folder/'job_timing.json').exists() else {}
        episodes=config.get('episodes',[])
        records=jsonlines(folder/'phase_separation.jsonl')
        hashes[run['label']]=jsonlines(folder/'phase_actions.jsonl')
        updates=[r for r in records if r['phase']=='observation_update' and r['status']=='encoded']
        sampling=log_records(folder/'server.log','[inference-profile]')
        cache=jsonlines(folder/'cache_memory.jsonl')
        cache_rows.extend(dict(label=run['label'],**r) for r in cache)
        tail=cache[-5:]
        plateau=(len(cache)>=36 and len({r['stored_entries'] for r in tail})==1
                 and all(r['observation_padding']==0 for r in tail))
        row=dict(retained_allocated_span_bytes=(max(r['allocated_bytes'] for r in tail)-min(r['allocated_bytes'] for r in tail) if tail else None),
                 cache_plateau_observed=plateau,cache_records=len(cache),label=run['label'],job_id=run.get('job_id'),exit_code=timing.get('exit_code'),
                 completed=any(e.get('status')=='completed' for e in episodes),
                 task_success=any(e.get('success') is True for e in episodes),
                 predictions=len(sampling),encoded_observation_updates=len(updates),
                 reached_32_chunk_window=len(updates)>=32,
                 transfer_seconds=sum(r.get('offload_seconds',0)+r.get('restore_seconds',0) for r in records),
                 encoding_seconds=sum(r.get('encoding_seconds',0) for r in records),
                 allocated_seconds=timing.get('allocated_seconds'))
        summaries.append(row)
        for r in records:
            events.append(dict(label=run['label'],**{k:v for k,v in r.items() if not isinstance(v,dict)},
                before_allocated_gib=r['before']['allocated_bytes']/1024**3,
                encoding_start_allocated_gib=r.get('encoding_start',{}).get('allocated_bytes',0)/1024**3,
                after_allocated_gib=r.get('after',{}).get('allocated_bytes',0)/1024**3))
    control=hashes.get('place_empty_cup_cache_control',[]);offload=hashes.get('place_empty_cup_cache_compact',[])
    parity=bool(control and offload and control==offload)
    cup_ok=all(r['completed'] and r['task_success'] and r['exit_code']==0 for r in summaries[:2])
    microwave=next((r for r in summaries if r['label'].startswith('open_microwave')),None)
    result=dict(runs=summaries,cup_action_hashes_identical=parity,
                cup_validation_passed=cup_ok,
                long_history_validated=bool(microwave and microwave['completed'] and microwave['exit_code']==0 and microwave['cache_plateau_observed']),
                ready_for_stage_1_to_4=False,
                note='Manual review required: this tests observation encoding, not memory during reward guidance. '
                     'A short successful microwave episode does not validate long history. '
                     'Action hashes are informational: compaction can change floating-point reductions. Inspect cup outcomes and videos.')
    (output/'stage00_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(output/'cache_memory.csv',cache_rows)
    write_csv(output/'phase_events.csv',events);write_csv(output/'phase_runs.csv',summaries)
    if events:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(10,11),layout='constrained')
        groups=defaultdict(list)
        for e in events:
            if e['phase']=='observation_update': groups[e['label']].append(e)
        for label,values in groups.items():
            axes[0].plot([r['chunk'] for r in values],[r['encoding_start_allocated_gib'] for r in values],label=label+' encoding')
            axes[0].plot([r['chunk'] for r in values],[r['after_allocated_gib'] for r in values],linestyle='--',label=label+' restored')
            axes[2].plot([r['chunk'] for r in values],
                [r.get('total_seconds',0) for r in values],label=label)
        axes[0].set(ylabel='Server PyTorch allocated GiB',title='Solid: weights offloaded for encoding; dashed: weights restored')
        for label in sorted({r['label'] for r in cache_rows}):
            rows=[r for r in cache_rows if r['label']==label]
            axes[1].plot([r['chunk'] for r in rows],[r['kv_and_metadata_bytes']/1024**3 for r in rows],label=label)
        axes[1].set(ylabel='Retained KV + metadata GiB',title='Check for a plateau beyond the history boundary')
        axes[2].set(xlabel='Chunk',ylabel='Encoding + transfers seconds')
        for ax in axes:ax.legend(fontsize=7)
        fig.savefig(output/'stage00_memory.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2));print(output)


if __name__=='__main__':main()
