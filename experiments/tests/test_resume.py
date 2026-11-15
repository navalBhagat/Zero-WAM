import json
from pathlib import Path

import pytest

from experiments.scripts.provenance import ROOT, snapshot
from experiments.scripts.robotwin_suite import extend_baselines, parser, specifications
from experiments.analysis.analyze import collect


def test_extend_pilot_preserves_scene_and_demo_offsets_and_analysis(tmp_path):
    manifest = tmp_path / 'pilot.json'
    a = parser().parse_args(['baselines', '--seeds', '0', '--episodes', '2',
                             '--spp', '8', '--manifest', str(manifest)])
    old = specifications(a)
    bundle = tmp_path / 'pilot.source.zip'
    snapshot(bundle)
    for index, spec in enumerate(old):
        spec['job_id'] = str(index)
        folder = Path(spec['environment']['EXPERIMENT_OUTPUT_ROOT']) / f'{spec["task"]}_{index}'
        folder.mkdir(parents=True)
        # Include a skipped simulator seed and an unsuccessful completed episode.
        episodes = [dict(episode_id=i, simulator_seed=seed, status='completed',
                         success=bool(i), icl_seed=0) for i, seed in enumerate([100000, 100003])]
        (folder / 'config.json').write_text(json.dumps(dict(episodes=episodes)))
        (folder / 'job_timing.json').write_text(json.dumps(dict(exit_code=0, allocated_seconds=10)))
    manifest.write_text(json.dumps(dict(stage='baselines', source_bundle=str(bundle), runs=old)))
    a.seeds = [0, 1, 2]; a.episodes = 10
    pending, reused = extend_baselines(specifications(a), manifest, ROOT)
    assert len(pending) == 12 and len(reused) == 4
    assert sum(int(s['environment']['TEST_NUM']) for s in pending) == 112
    for spec in pending:
        env = spec['environment']
        if spec['seed'] == 0:
            assert env['EPISODE_OFFSET'] == '2' and env['START_SEED'] == '100004'
            assert env['TEST_NUM'] == '8'
        else:
            assert 'START_SEED' not in env and env['TEST_NUM'] == '10'
    _, episodes, _, _ = collect(dict(stage='baselines', runs=reused + pending,
                                     estimate=dict(sbu_per_hour=128)), None)
    assert len(episodes) == 8 and sum(e['success'] for e in episodes) == 4
    # Changed rendering must not silently reuse a different experiment.
    a.spp = 16
    fresh, reused = extend_baselines(specifications(a), manifest, ROOT)
    assert not reused and sum(int(s['environment']['TEST_NUM']) for s in fresh) == 120
    # Gaps cannot be continued safely by a single episode offset.
    folder = Path(old[0]['environment']['EXPERIMENT_OUTPUT_ROOT']) / f'{old[0]["task"]}_0'
    config = json.loads((folder / 'config.json').read_text())
    config['episodes'][1]['episode_id'] = 2
    (folder / 'config.json').write_text(json.dumps(config))
    a.spp = 8
    with pytest.raises(ValueError, match='gaps'):
        extend_baselines(specifications(a), manifest, ROOT)
