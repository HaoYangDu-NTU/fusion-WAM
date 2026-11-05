"""Batched VPP2 cached-video Euler inference, preserving each request's seeds."""
import torch


@torch.inference_mode()
def infer_action_batch(model, images, proprio, context, context_mask):
    batch = images.shape[0]
    if images.ndim != 4 or images.shape[1] != 3 or batch < 1:
        raise ValueError("Expected nonempty [B,3,H,W] images")
    if model.action_chunk_size != 1 or model.proprio_dim != 8:
        raise ValueError("Expected the released LIBERO action/state dimensions")
    if (model.action_clean_first_frame_start_layer is not None
            or model.action_clean_first_frame_timestep_cutoff is not None):
        raise ValueError("Experimental first-frame routing is not supported")
    images = images.to(device=model.device, dtype=model.torch_dtype)
    # The native VAE resets its temporal cache independently for each image.
    condition = model._encode_video_latents(images.unsqueeze(2), tiled=False)
    clip = model._encode_clip_fea_from_image(images)
    context = context.to(device=model.device, dtype=model.torch_dtype)
    context_mask = context_mask.to(device=model.device, dtype=torch.bool)
    context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)
    actions = torch.randn((1, 32, model.action_expert.action_dim),
        generator=torch.Generator(device="cpu").manual_seed(42), dtype=torch.float32)
    actions = actions.to(device=model.device, dtype=model.torch_dtype).expand(batch, -1, -1).clone()
    # Each native request draws from fresh action42/video43 generators.
    video = model._sample_action_video_latents(first_frame_latents=condition[:1],
        num_video_frames=17, generator=torch.Generator(device="cpu").manual_seed(43), rand_device="cpu")
    video = video.expand(batch, -1, -1, -1, -1).clone()
    timestep_video = torch.full((batch,), float(model.train_video_scheduler.num_train_timesteps),
        device=model.device, dtype=condition.dtype)
    cache, attention_mask, video_seq_len = model._build_frozen_video_cache(
        latents=video, timestep_video=timestep_video, context=context, context_mask=context_mask,
        condition_latents=condition, clip_fea=clip, action_seq_len=32,
        fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
        return_video_prediction=False, max_video_layers=None)
    times, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=10, device=model.device, dtype=actions.dtype, shift_override=5.)
    for timestep, delta in zip(times, deltas):
        prediction = model._predict_action_noise_with_cache(
            latents_action=actions, timestep_action=timestep.expand(batch), context=context,
            context_mask=context_mask, video_kv_cache=cache, attention_mask=attention_mask,
            video_seq_len=video_seq_len)
        actions = model.infer_action_scheduler.step(prediction, delta, actions)
    return model._unpack_action_tokens(actions).float().cpu()
