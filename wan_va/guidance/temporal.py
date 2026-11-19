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


class DistanceFrameSelector(WindowFrameSelector):
    """Same greedy alignment, with a bounded reference visual-distance horizon.

    Distances are cosine changes between smoothed whole-image DINO embeddings,
    not a semantic completion estimate. Matching never advances to the target.
    """
    def __init__(self, features, window=24, lookahead=4, distance=None,
                 smoothing=3, max_frames=24):
        import bisect
        import statistics
        import torch
        if smoothing < 1 or smoothing % 2 != 1 or max_frames < 1:
            raise ValueError('Smoothing must be positive odd; max_frames must be positive')
        if distance is not None and (not math.isfinite(distance) or distance <= 0):
            raise ValueError('Distance increment must be finite and positive')
        if features.ndim != 2 or not len(features) or not torch.isfinite(features).all():
            raise ValueError('Expected finite nonempty [frames, features] embeddings')
        super().__init__(len(features), window, lookahead)
        features = features.detach().float().cpu()
        radius = smoothing // 2
        smoothed = torch.stack([features[max(0, i-radius):i+radius+1].mean(0)
                                for i in range(len(features))])
        smoothed = torch.nn.functional.normalize(smoothed, dim=-1)
        changes = (1 - (smoothed[1:] * smoothed[:-1]).sum(-1)).clamp(0, 2)
        # Float32 dot-product noise must not turn an identical reference into motion.
        changes[changes < 1e-6] = 0
        self.coordinate = [0.] + changes.cumsum(0).tolist()
        spans = [self.coordinate[i+lookahead]-self.coordinate[i]
                 for i in range(max(0, len(features)-lookahead))] if lookahead else []
        self.distance = distance if distance is not None else (statistics.median(spans) if spans else 0.)
        self.max_frames, self.smoothing = max_frames, smoothing
        self.calibration = 'explicit' if distance is not None else 'reference_median_frame_horizon'
        self.details = {}
        self._bisect = bisect.bisect_left

    def select(self, chunk_index, frame_indices=None, similarities=None):
        matched = super().select(chunk_index, frame_indices, similarities)
        if frame_indices is not None:
            self.details = {}
            return matched
        start = self.position
        limit = min(self.frame_count-1, start+self.max_frames)
        if self.distance <= 1e-12:
            # No positive calibrated horizon: preserve the frame baseline, bounded.
            target = min(limit, start+self.lookahead)
            fallback = 'zero_calibrated_distance'
        else:
            target = min(limit, self._bisect(self.coordinate,
                         self.coordinate[start]+self.distance, lo=start))
            fallback = None
        actual = self.coordinate[target]-self.coordinate[start]
        self.details = dict(lookahead_mode='dino_distance', target_frame=target,
            matched_distance=self.coordinate[start], target_distance=self.coordinate[target],
            requested_distance=self.distance, actual_distance=actual,
            frame_offset=target-start, horizon_limited=actual+1e-12 < self.distance,
            max_forward_frames=self.max_frames, smoothing=self.smoothing,
            distance_calibration=self.calibration, fallback=fallback)
        return [target]


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
        match_details.update(getattr(server.guidance_selector, 'details', {}))
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
