# Outputs

All paths are below the campaign directory chosen at planning time:

```text
manifest.json, source.zip
jobs/JOB/settings.json, slurm_JOBID.out, summary.json
jobs/JOB/diagnostics/{client.log,server.log,gpu.csv,job_timing.json,...}
trials/TASK/scene_SLOT_noise_SEED/METHOD/{config.json,results.csv,videos,diagnostics}
trials/TASK/scene_SLOT_noise_SEED/METHOD/noise/{demo.pt,initial.pt,history,pairs,chunks.jsonl}
pairs/PAIR_ID/{record.json,pairs/chunk_NNNN.pt}       # fixed-history collection
jobs/train/{adapter.pt,latest.pt,training.jsonl}      # stage 5
trials.csv, success.csv, timing.csv, analysis.json, collection.json, success.png
```

`adapter.pt` is the best validation checkpoint; `latest.pt` includes optimizer and
RNG state for resume. Initial Gaussian/recovered noise is saved for every oracle
chunk. Collection contexts refer to shared demo/history files instead of copying
all histories per seed. Partial attempts are archived, not combined with new files.

Historical campaign directories are unchanged. They retain their old schema and
scientific settings; they are not automatically migrated or merged into the new
caption-free online-oracle analysis.
