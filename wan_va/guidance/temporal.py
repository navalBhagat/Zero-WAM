"""Opt-in whole-image temporal matching for Robotwin experiments."""
import math
import operator


class WindowFrameSelector:
    """Monotonic local visual alignment; target may look ahead of the match."""
    def __init__(self, frame_count, window=24, lookahead=0):
        if frame_count < 1 or window < 1 or lookahead < 0:
            raise ValueError('Invalid temporal selector settings')
        self.frame_count, self.window, self.lookahead = frame_count, window, lookahead
        self.position = 0
        self.fixed_frames = [0]  # Initial preview only; select updates every chunk.

    def select(self, chunk_index, frame_indices=None, similarities=None):
        if frame_indices is not None:
            if not frame_indices or any(isinstance(i, bool) for i in frame_indices):
                raise ValueError('Select integer frame indices')
            indices = [operator.index(i) for i in frame_indices]
            if len(set(indices)) != len(indices) or any(i < 0 or i >= self.frame_count for i in indices):
                raise ValueError('Invalid explicit frame indices')
            return indices
        if similarities is None or len(similarities) != self.frame_count:
            raise ValueError('One similarity score per reference frame is required')
        if not all(math.isfinite(float(v)) for v in similarities):
            raise ValueError('Similarity scores must be finite')
        stop = min(self.frame_count, self.position + self.window + 1)
        candidates = range(self.position, stop)
        self.position = max(candidates, key=lambda i: float(similarities[i]))
        return [min(self.frame_count - 1, self.position + self.lookahead)]


def select_reference(server, request):
    """Match latest observed external-camera RGB, freeze target and write a QA trace."""
    import json
    import os
    import time
    import torch
    explicit = request.get("guidance_reference_frames")
    selection_started = time.perf_counter()
    match_details = {}
    if isinstance(server.guidance_selector, WindowFrameSelector) and explicit is None:
        image = getattr(server, 'latest_guidance_observation', None)
        if image is None:
            image = request['obs']
        if isinstance(image, list):
            image = image[-1]
        image = torch.from_numpy(image[server.job_config.obs_cam_keys[0]].copy())
        image = image.permute(2, 0, 1).unsqueeze(0).to(server.device).float() / 255
        with torch.no_grad():
            query = server.external_reward.features(image)[0]
            scores = (server.external_reward.reference_features @ query).cpu().tolist()
        indices = server.guidance_selector.select(server.chunk_idx, similarities=scores)
        match_details = dict(matched_frame=server.guidance_selector.position,
                             matched_similarity=scores[server.guidance_selector.position])
        from PIL import Image
        preview = image[0].detach().cpu().permute(1, 2, 0).mul(255).byte().numpy()
        preview_path = os.path.join(server.reference_preview_dir, f"robot_chunk_{server.chunk_idx:04d}.png")
        Image.fromarray(preview).save(preview_path)
        match_details['robot_preview_path'] = preview_path
    else:
        indices = server.guidance_selector.select(server.chunk_idx, explicit)
    server.external_reward.select_reference(indices)
    selection = dict(chunk=server.chunk_idx, frame_indices=indices,
                     selection=(os.environ.get("GUIDANCE_SELECTION", "fixed") if explicit is None
                                else "explicit_request"),
                     selection_seconds=time.perf_counter() - selection_started,
                     preview_paths=server.external_reward.save_reference_frames(server.reference_preview_dir))
    selection.update(match_details)
    selection['selection_seconds'] = time.perf_counter() - selection_started
    with open(os.path.join(server.reference_preview_dir, "selection.jsonl"), "a") as stream:
        stream.write(json.dumps(selection) + "\n")
    server.last_guidance_selection = selection
    import logging
    logging.getLogger(__name__).info("[guidance-reference] %s", selection)
    return selection
