"""Deterministic caption retrieval over local HumanGen latent files (CPU only).

BM25 ranks caption/description text; it does not verify video semantics. Cached
metadata is read on CPU with weights_only=True; no decoding or GPU model is used.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path


def tokens(text):
    text = text.lower().replace('electronicscale', 'electronic scale').replace('breadbasket', 'bread basket')
    aliases = {'breads': 'bread', 'loaf': 'bread', 'loaves': 'bread', 'placing': 'place',
               'put': 'place', 'lay': 'place', 'set': 'place', 'grab': 'grasp', 'take': 'grasp',
               'both': 'two', 'single': 'one'}
    stop = {'the', 'a', 'an', 'to', 'of', 'on', 'in', 'into', 'and', 'with', 'using',
            'use', 'it', 'there', 'is', 'if', 'robotwin', 'generated', 'video', 'kling', 'v3'}
    return [aliases.get(t, t) for t in re.findall(r'[a-z]+', text) if t not in stop]


def description(path):
    # HumanGen stores its real caption inside the latent file, not its folder name.
    import torch
    try:
        item = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error).lower():
            raise
        item = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(item, dict):
        for key in ('text', 'local_instruction', 'task'):
            if isinstance(item.get(key), str) and item[key].strip():
                return item[key], f'{path}:{key}'
    for name in ('metadata.json', 'caption.json', 'prompt.json'):
        sidecar = path.parent / name
        if sidecar.is_file():
            obj = json.loads(sidecar.read_text())
            if isinstance(obj, str) and obj.strip():
                return obj, str(sidecar)
            if isinstance(obj, dict):
                for key in ('human_caption', 'caption', 'task_description', 'instruction', 'prompt'):
                    if isinstance(obj.get(key), str) and obj[key].strip():
                        return obj[key], f'{sidecar}:{key}'
    return re.sub(r'^\d+_robotwin_', '', path.parent.name).replace('_', ' '), 'sample_directory_description'


def bread_count(caption):
    """Conservative short-caption object count; arm/hand counts are not object counts."""
    text = caption.lower()
    if re.search(r'\b(two|both)\s+(?:[\w-]+\s+){0,3}(breads|loaves)\b', text):
        return 2
    if re.search(r'\b(bread|loaf)\b[^.!?]*\band\s+(a|an)\s+(?:[\w-]+\s+){0,3}(bread|loaf)\b', text):
        return 2
    if re.search(r'\b(one|single|a|an)\s+(?:[\w-]+\s+){0,3}(bread|loaf)\b', text):
        return 1
    return None


class DemonstrationRetriever:
    def __init__(self, latent_root):
        self.root = Path(latent_root)
        self.items = []
        for path in sorted(self.root.rglob('generated_video*.pth')):
            caption, source = description(path)
            words = tokens(caption)
            if words:
                self.items.append(dict(latent=str(path.resolve()), caption=caption,
                                       caption_source=source, words=words,
                                       domain_words=set(tokens(path.parent.name)) | set(words),
                                       caption_object_count=bread_count(caption)))
        if not self.items:
            raise ValueError(f'No generated_video*.pth demonstrations found in {self.root}')

    def select(self, task, instruction, object_count=None):
        # Restrict to the task's named manipulated-object domain before text ranking.
        anchors = {'place_object_scale': {'scale'}, 'place_bread_basket': {'bread'}}
        if task not in anchors:
            raise ValueError(f'Automatic retrieval is not configured for task {task}')
        candidates = [i for i in self.items if anchors[task] & i['domain_words']]
        if not candidates:
            raise ValueError(f'No caption candidates for {task}; inspect the local HumanGen inventory')
        query = set(tokens(instruction))
        if object_count in (1, 2):
            query.discard('one' if object_count == 2 else 'two')
            query.add('one' if object_count == 1 else 'two')
        documents = [Counter(i['words']) for i in candidates]
        average = sum(sum(d.values()) for d in documents) / len(documents)
        frequency = Counter(word for d in documents for word in d)
        scores = []
        for item, counts in zip(candidates, documents):
            score = 0.
            for word in query:
                n = counts[word]
                idf = math.log(1 + (len(documents) - frequency[word] + .5) / (frequency[word] + .5))
                score += idf * n * 2.5 / (n + 1.5 * (.25 + .75 * sum(counts.values()) / average))
            desired_count = object_count if object_count in (1, 2) else bread_count(instruction)
            if task == 'place_bread_basket' and desired_count and item['caption_object_count'] and item['caption_object_count'] != desired_count:
                score -= 2.
            scores.append((score, item))
        ranked = sorted(scores, key=lambda pair: (-pair[0], pair[1]['latent']))
        score, selected = ranked[0]
        def audit(pair):
            value, item = pair
            return {k: item[k] for k in ('latent', 'caption', 'caption_source', 'caption_object_count')} | {'score': value}
        record = dict(method='bm25_caption', task=task, query=instruction,
                      scene_object_count=object_count,
                      candidate_count=len(candidates), indexed_count=len(self.items),
                      selected=audit(ranked[0]), top_candidates=[audit(p) for p in ranked[:5]],
                      limitation='Text retrieval; video contents and actual object count are not verified')
        latent = Path(selected['latent'])
        # A cached latent suffices for guidance/ICL. Derive the MP4 provenance path.
        parts = list(latent.parts)
        if 'human_latents' in parts:
            parts[parts.index('human_latents')] = 'human_data'
        video = str(Path(*parts).with_suffix('.mp4'))
        return video, str(latent), record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('latent_root', type=Path)
    parser.add_argument('--task', required=True, choices=['place_bread_basket', 'place_object_scale'])
    parser.add_argument('--instruction', required=True)
    options = parser.parse_args()
    _, _, audit = DemonstrationRetriever(options.latent_root).select(options.task, options.instruction)
    print(json.dumps(audit, indent=2))
