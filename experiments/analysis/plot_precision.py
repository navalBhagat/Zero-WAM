"""Plot the paired captured-step numerical diagnostic; no GPU required."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import csv


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path); a=p.parse_args()
    data=json.loads(a.input.read_text()); output=a.output or a.input.parent; output.mkdir(parents=True,exist_ok=True)
    rows=data['measurements']; keys=list(dict.fromkeys(k for r in rows for k in r))
    with (output/'precision.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    groups=defaultdict(list)
    for r in rows:
        if r['status']=='ok': groups[(r['decoder_dtype'],r['strength'])].append(r)
    for dtype in ('float32','float16'):
        strengths=sorted(s for d,s in groups if d==dtype)
        for ax,field,label in zip(axes.flat,
            ('reward_gain','applied_correction_norm','changed_fraction','seconds'),
            ('Local reward gain','Applied correction L2','Fraction of editable elements changed','Replay seconds (incl. score check)')):
            ys=[statistics.median(r[field] for r in groups[(dtype,s)]) for s in strengths]
            ax.plot(strengths,ys,'o-',label=dtype); ax.set(xscale='log',xlabel='Guidance strength',ylabel=label)
            ax.legend()
    failed=[r for r in rows if r['status']!='ok']
    fig.suptitle(f'Reward decoder precision; FP32 state and DINO. Failed trials: {len(failed)}/{len(rows)}')
    fig.savefig(output/'precision.png',dpi=180); fig.savefig(output/'precision.pdf'); plt.close(fig)
    (output/'failures.json').write_text(json.dumps(failed,indent=2)+'\n')
    print(f'{output}/precision.png; failures: {len(failed)}. This is not a task-success measurement.')


if __name__=='__main__': main()
