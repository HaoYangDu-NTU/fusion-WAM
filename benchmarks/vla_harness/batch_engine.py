"""Batched Wan TI2V deployment preserving the native joint denoising equations.

Reference: BaseWAMArchitecture.generate. Supported path: external encoders, sync
joint denoising, CFG=1, action-only evaluation. Training and official single-
sample generation are untouched. Cache decisions remain independent per sample.
"""

from __future__ import annotations

import numpy as np
import torch

from openwam.deploy.denoise_schedule import make_schedule
from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
from openwam.model.architectures.utils.flow_prediction import prediction_to_velocity


def _select_inputs(inputs, indices, batch_size):
    return {
        k: v.index_select(0, indices) if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == batch_size else v
        for k, v in inputs.items()
    }


class BatchedOpenWAMEngine:
    def __init__(self, native_engine):
        self.native = native_engine
        self.architecture = native_engine.architecture
        self.cfg = native_engine.cfg
        self.architecture.eval()
        vb = self.architecture.video_backbone
        if vb.video_encoder is None or not vb._is_ti2v or vb._has_vace:
            raise ValueError("Batch adapter requires an external-encoder Wan TI2V checkpoint")
        if native_engine._cfg_scale != 1.0 or self.cfg.inference.denoise_mode != "sync":
            raise ValueError("Batch adapter requires the official LIBERO sync / CFG=1 protocol")
        if native_engine._decode_video:
            raise ValueError("Latent encoder evaluation requires optimization.decode_video=false")
        self.last_stats = {}

    def _new_cache(self):
        source = self.native._dit_cache
        if source is None:
            return None
        return DiTVelocityCache(
            cosine_threshold=source.cosine_threshold,
            max_consecutive_skips=source.max_consecutive_skips,
            interpolation_weight=source.interpolation_weight,
        )

    def _prepare(self, observations):
        arch, inf = self.architecture, self.cfg.inference
        vb = arch.video_backbone
        action_frames = int(inf.num_frames)
        video_frames = int(getattr(inf, "video_num_frames", action_frames))
        shift = getattr(arch.action_backbone, "shift_action", None)
        shift = 5.0 if shift is None else shift
        schedule = make_schedule(
            "sync",
            video_scheduler=arch.video_scheduler,
            action_scheduler=arch.action_scheduler,
            num_steps=inf.denoise_steps,
            shift=shift,
            shift_video=vb.shift_video,
        )
        # Reuse native text/noise/grid prep. Encode all reference images together
        # below; a list of B images must not become a video of length B.
        prepared = []
        for obs in observations:
            prepared.append(vb.preprocess_input_for_inference(
                prompt=obs["prompt"],
                clean_condition_images=None,
                num_frames=video_frames,
                height=inf.height,
                width=inf.width,
                seed=42,
                num_inference_steps=inf.denoise_steps,
                shift=shift,
                tiled=True,
                vace_cache=None,
                prompt_embed_cache=self.native._prompt_embed_cache,
                # Every request has the same seed, grid and noise dtype. Reuse
                # the initial tensor; cat/clone below keep denoising independent.
                initial_noise=prepared[0]["noise"] if prepared else None,
            ))
        inputs = dict(prepared[0])
        for key in ("context", "seq_lens", "noise", "latents"):
            inputs[key] = torch.cat([p[key] for p in prepared], dim=0)
        encoder = vb.video_encoder
        clips = [[obs["image"]] for obs in observations]
        history_num_tubelets = getattr(encoder, "history_num_tubelets", 0)
        frame_offsets = getattr(encoder, "clean_condition_frame_offsets", None)
        if frame_offsets is not None:
            clips = [obs["clean_condition_images"] for obs in observations]
            if any(len(clip) != len(frame_offsets) for clip in clips):
                raise ValueError("Wrong number of video encoder history/current frames")
        elif history_num_tubelets:
            clips = [obs["clean_condition_images"] for obs in observations]
            if any(len(clip) != 2 * history_num_tubelets for clip in clips):
                raise ValueError("clean_condition_images must contain 2 * history_num_tubelets images")
        elif getattr(encoder, "add_one_history_context", False):
            clips = [[obs["history_image"], obs["image"]] for obs in observations]
        pixels = encoder.preprocess_video_batch(clips)
        ref = vb.encode_video(pixels).to(device=arch.device, dtype=arch.dtype)
        inputs.update(clean_condition_latents=ref, fuse_vae_embedding_in_latents=True, num_clean_condition_latents=ref.shape[2])
        inputs["latents"] = inputs["latents"].clone()
        inputs["latents"][:, :, : ref.shape[2]] = ref
        if arch.uses_proprioception:
            inputs["proprio"] = torch.stack(
                [
                    arch.normalize_deploy_proprio(obs["state"]).to(device=arch.device, dtype=arch.dtype)
                    for obs in observations
                ]
            )
        # Each native request restarts its own generator at seed=42. Drawing
        # (B,T,D) at once would change the noise of samples 1..B-1.
        noise = torch.randn(
            1,
            action_frames - 1,
            arch.action_dim,
            device=arch.device,
            dtype=arch.dtype,
            generator=torch.Generator(device=arch.device).manual_seed(42),
        )
        actions = noise.expand(len(observations), -1, -1).clone()
        return inputs, actions, schedule

    @torch.no_grad()
    def generate_batch(self, observations):
        if not observations:
            return []
        arch = self.architecture
        inputs, actions, schedule = self._prepare(observations)
        batch_size = len(observations)
        caches = [self._new_cache() for _ in observations]
        inactive = arch._resolve_inactive_action_dims(None, arch.device)
        inactive_noise = None
        forward_sizes = []
        for i in range(len(schedule) - 1):
            tv, ta = schedule[i]
            tv_next, ta_next = schedule[i + 1]
            sv = tv / float(arch.video_scheduler.num_train_timesteps)
            sa = ta / float(arch.action_scheduler.num_train_timesteps)
            sv_next = tv_next / float(arch.video_scheduler.num_train_timesteps)
            sa_next = ta_next / float(arch.action_scheduler.num_train_timesteps)
            video_step, action_step = sv != sv_next, sa != sa_next
            if not video_step and not action_step:
                continue
            compute = [
                j
                for j, cache in enumerate(caches)
                if cache is None or not video_step or cache.should_recompute(sv, require_action=action_step)
            ]
            video_predictions, action_predictions = {}, {}
            if compute:
                if len(compute) == batch_size:
                    selected, selected_actions = inputs, actions
                else:
                    indices = torch.tensor(compute, device=arch.device)
                    selected = _select_inputs(inputs, indices, batch_size)
                    selected_actions = actions.index_select(0, indices)
                torch.compiler.cudagraph_mark_step_begin()
                video_pred, action_pred = arch.forward(
                    selected_actions,
                    torch.full((len(compute),), ta, device=arch.device, dtype=arch.dtype),
                    **selected,
                    timestep=torch.full((len(compute),), tv, device=arch.device, dtype=arch.dtype),
                )
                if action_pred is None:
                    raise RuntimeError("OpenWAM LIBERO checkpoint did not predict actions")
                video_pred = prediction_to_velocity(
                    video_pred, selected["latents"], sv,
                    getattr(arch, "video_prediction_type", "v"),
                ).to(dtype=video_pred.dtype)
                action_pred = prediction_to_velocity(
                    action_pred, selected_actions, sa,
                    getattr(arch, "action_prediction_type", "v"),
                ).to(dtype=action_pred.dtype)
                forward_sizes.append(len(compute))
                for row, j in enumerate(compute):
                    video_predictions[j] = video_pred[row : row + 1]
                    action_predictions[j] = action_pred[row : row + 1]
                    if caches[j] is not None and video_step:
                        caches[j].update(video_predictions[j], sv, action_predictions[j])
            for j in range(batch_size):
                if j not in video_predictions:
                    video_predictions[j] = caches[j].get_cached()
                    action_predictions[j] = caches[j].get_cached_action()
            if video_step:
                prediction = video_pred if len(compute) == batch_size else torch.cat([video_predictions[j] for j in range(batch_size)], dim=0)
                latents = inputs["latents"] + prediction * (sv_next - sv)
                ref = inputs.get("clean_condition_latents")
                if ref is not None:
                    latents = latents.clone()
                    latents[:, :, : ref.shape[2]] = ref
                inputs["latents"] = latents
            if action_step:
                prediction = action_pred if len(compute) == batch_size else torch.cat([action_predictions[j] for j in range(batch_size)], dim=0)
                if inactive is not None and inactive_noise is None:
                    if float(sa) <= 0:
                        raise ValueError("Cannot initialize inactive action noise from non-positive sigma")
                    inactive_noise = actions[..., inactive].detach().clone() / float(sa)
                actions = arch.action_scheduler.flow_step(prediction, sa, sa_next, actions)
                if inactive is not None:
                    actions[..., inactive] = inactive_noise * float(sa_next)
        chunks = []
        normalizer = getattr(arch, "normalizer", None)
        action_scale = getattr(arch, "action_scale", 1.0)
        for chunk in actions.float().cpu().numpy():
            if action_scale != 1.0:
                chunk = chunk / action_scale
            if normalizer is not None:
                chunk = normalizer.unnormalize(chunk)
            # The same final command projection as WAMPolicy.
            for dim in getattr(arch, "binary_command_dims", ()) or ():
                chunk[..., dim] = np.where(chunk[..., dim] > 0.5, 1.0, -1.0)
            chunks.append(chunk)
        self.last_stats = {
            "requests": batch_size,
            "forward_batch_sizes": forward_sizes,
            "cache": [cache.stats if cache else None for cache in caches],
        }
        return chunks
