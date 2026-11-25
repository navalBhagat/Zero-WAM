"""Canonical HumanGen latent layouts."""
import torch

def reshape_latent(latent, item):
    """Match the released server's supported HumanGen latent layouts."""
    if not torch.is_tensor(latent):
        raise ValueError("ICL latent must be a tensor")
    layout = str(item.get("latent_layout", "")).lower().replace("_", " ")
    if latent.ndim == 5:
        if latent.shape[0] != 1:
            raise ValueError(f"ICL latent batch must be 1, got {latent.shape}")
        return latent.contiguous()
    if latent.ndim == 4:
        if layout in ("f h w c", "f h w d"):
            return latent.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        return latent.unsqueeze(0).contiguous()
    if latent.ndim == 2:
        f, h, w = (int(item[key]) for key in
                   ("latent_num_frames", "latent_height", "latent_width"))
        if latent.shape[0] != f * h * w:
            raise ValueError(f"Flattened ICL latent {latent.shape} does not match {(f, h, w)}")
        return latent.view(f, h, w, -1).permute(3, 0, 1, 2).unsqueeze(0)
    raise ValueError(f"Unsupported ICL latent shape: {latent.shape}")
