import torch


class CausalPatchMask:
    def __init__(
        self,
        image_size,
        patch_size,
        tubelet_size,
        video_frames,
        context_tubelets=1,
        num_views=1,
    ):
        height, width = image_size
        grid_depth = video_frames // tubelet_size
        if num_views <= 0:
            raise ValueError(f"num_views must be positive, got {num_views}")
        grid_height = height // patch_size
        grid_width = width // patch_size
        n_ctx = context_tubelets * num_views * grid_height * grid_width
        n_tot = grid_depth * num_views * grid_height * grid_width
        if not 0 < n_ctx < n_tot:
            raise ValueError(
                f"context_tubelets={context_tubelets} invalid for grid_depth={grid_depth}"
            )
        token_ids = torch.arange(n_tot, dtype=torch.long).reshape(
            grid_depth, num_views, grid_height, grid_width
        )
        self.ctx_idx = token_ids[:context_tubelets].flatten()
        self.tgt_idx = token_ids[context_tubelets:].flatten()
        self.n_ctx = self.ctx_idx.numel()
