"""Visual QA for temporal matching: trajectories, overhead, robot/reference contact sheets."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path); p.add_argument('--panels',type=int,default=12)
    a=p.parse_args()
    if a.panels<1: p.error('--panels must be positive')
    output=a.output or a.run/'matching_analysis'; output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    paths=sorted(a.run.glob('references/episode_*/selection.jsonl'))
    if not paths: p.error('No selection traces found; use a guided stage-3 run')
    for path in paths:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows: continue
        name=path.parent.name
        fig,axes=plt.subplots(3,1,figsize=(9,8),layout='constrained')
        x=[r['chunk'] for r in rows]
        axes[0].plot(x,[r.get('matched_frame',r['frame_indices'][0]) for r in rows],label='Matched frame')
        axes[0].plot(x,[r['frame_indices'][0] for r in rows],label='Guidance target'); axes[0].legend()
        axes[0].set_ylabel('Human RGB frame index')
        axes[1].plot(x,[r.get('matched_similarity',float('nan')) for r in rows]); axes[1].set_ylabel('Cosine similarity')
        axes[2].plot(x,[r['selection_seconds']*1000 for r in rows]); axes[2].set(xlabel='Robot chunk',ylabel='Selection + preview ms')
        fig.suptitle(name+' — check for stalls and premature advances'); fig.savefig(output/f'{name}_matching.png',dpi=170); plt.close(fig)
        chosen=sorted(set(round(i*(len(rows)-1)/max(1,min(a.panels,len(rows))-1))
                          for i in range(min(a.panels,len(rows)))))
        sheet=Image.new('RGB',(640,205*len(chosen)), 'white'); draw=ImageDraw.Draw(sheet)
        for line,index in enumerate(chosen):
            row=rows[index]; y=line*205
            robot=path.parent/Path(row.get('robot_preview_path','missing')).name
            human=path.parent/f'frame_{row["frame_indices"][0]:04d}.png'
            for column,image_path in enumerate((robot,human)):
                if image_path.is_file():
                    with Image.open(image_path) as image:
                        image=image.convert('RGB'); image.thumbnail((315,175)); sheet.paste(image,(column*320,y+25))
            draw.text((5,y+5),f'Robot chunk {row["chunk"]}',fill='black')
            draw.text((325,y+5),f'Human target frame {row["frame_indices"][0]}',fill='black')
        sheet.save(output/f'{name}_contact_sheet.jpg')
    print(output)


if __name__=='__main__': main()
