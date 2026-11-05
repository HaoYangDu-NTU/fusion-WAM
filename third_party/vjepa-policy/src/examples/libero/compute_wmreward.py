# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare predicted-future-latent vs. GT-future-latent similarity across three
world models on the SAME sampled LIBERO episodes:

  1. V-JEPA 2      (vjepa2_vit_giant, 256)            -- pretrained predictor
  2. V-JEPA 2.1    (vjepa2_1_vit_giant_384, 384)      -- hierarchical predictor
  3. Ours          (frozen vjepa2 vit-giant encoder + our T5-conditioned FiLM
                    predictor, trained by langwam)     -- language-conditioned

All models use the identical eval protocol as ``compute_wmreward_libero.py``:
the predictor sees only the first tubelet (2 frames of context) and predicts the
future frames of each 16-frame window; the window slides with stride 15; the
score per window is ``1 - cosine_similarity(predicted, GT-future)`` (and the
persistence / intra-suite-shuffle baselines). Ours additionally receives the
episode's task instruction as a precomputed T5-XXL embedding (FiLM), matching how
it was trained.

Every model is scored on the SAME episodes and the SAME shuffle-donor episodes
(sampled once). V-JEPA 2 and Ours share one frozen encoder, so the ONLY
difference between them is the predictor -- isolating what the language-
conditioned fine-tune adds. Numbers are reported in the last-layer 1408-d cosine
space so all three are directly comparable (2.1 also natively lives in a 5632-d
hierarchical space; we report its 1408-d slice here).

Usage:
    python compute_wmreward_libero_langvjepa.py --n_episodes 10 \
        --lang_ckpt runs/langwam_vitg_libero4_wm_t5xxl/checkpoint_step008460.pt \
        --text_cache_dir ./data/text_embeds/libero_t5xxl --lang_dim 4096
"""

import argparse
import hashlib
import os
import random
import sys
from collections import defaultdict

import torch

# This world-model SIMILARITY eval reuses the heavy V-JEPA eval infrastructure
# (sliding-window loss, torch-hub 2.0/2.1 loaders, LIBERO video decoding) that
# lives in the VJEPA-WAM repo -- it is intentionally NOT vendored here (it is an
# evaluation stack, not part of the training package). Point VJEPA_WAM_ROOT at
# that checkout and run this in an env with decord + AV1 (the `VLA_JEPA` env),
# NOT `prts-ft`. Only the MODEL definition is imported from this package.
VJEPA_WAM_ROOT = os.environ.get("VJEPA_WAM_ROOT")
if not VJEPA_WAM_ROOT:
    raise RuntimeError("Set VJEPA_WAM_ROOT to a VJEPA-WAM checkout before evaluation")
if VJEPA_WAM_ROOT not in sys.path:
    sys.path.insert(0, VJEPA_WAM_ROOT)

# Reuse the exact video loader / suite discovery / instruction mapping used by the
# 2.0 and 2.1 scripts so preprocessing and episode sampling are identical.
from compute_wmreward import load_vjepa_models
from compute_wmreward_libero import (
    DATA_ROOT,
    SUITES,
    episode_index_from_path,
    find_episode_videos,
    load_episode_instructions,
    load_video_as_tensor,
)
from compute_wmreward_libero_v21 import (
    compute_vjepa21_loss_sliding_window,
    load_vjepa21_giant,
)
from utils import compute_vjepa_loss_sliding_window

# The dataset defines the exact instruction template used at train time and is
# what makes the T5 cache filenames match.
from vjepa_policy.datasets import DEFAULT_PROMPT
from vjepa_policy.models import vision_transformer as video_vit
from vjepa_policy.models.lang_predictor import lang_vit_predictor

_VITG_CKPT = os.environ.get("VJEPA2_ENCODER_CHECKPOINT")


def build_training_encoder(device, img_size=256, num_frames=16):
    """Frozen encoder built EXACTLY as langwam training: load ``vitg.pt['encoder']``
    (the ONLINE encoder). This is critical: the hub ``vjepa2_vit_giant`` loads
    ``vitg.pt['target_encoder']`` (the EMA encoder) instead, which differs by
    ~0.04/param and yields different features (cos~0.26). Our predictor was trained
    on the online encoder, so it must be evaluated on the same weights (the
    pretrained predictor is robust to the online/EMA swap, but ours overfit to it).
    (uniform_power is a no-op here: under use_rope=True there is no sincos pos-embed.)"""
    if not _VITG_CKPT:
        raise RuntimeError("Set VJEPA2_ENCODER_CHECKPOINT before evaluation")
    ck = torch.load(_VITG_CKPT, map_location="cpu", weights_only=False)
    sd = {k[len("module.backbone."):]: v for k, v in ck["encoder"].items()
          if k.startswith("module.backbone.")}
    enc = video_vit.vit_giant(img_size=img_size, patch_size=16, num_frames=num_frames,
                              tubelet_size=2, uniform_power=False, use_rope=True, use_sdpa=True)
    enc.load_state_dict(sd)
    return enc.to(device).eval()


# ----------------------------- shared eval plan ----------------------------- #
def build_eval_plan(data_root, view, n_episodes, n_donor, seed):
    """Sample scored episodes + shuffle donors ONCE, shared across all models.

    Reproduces the sampling of compute_wmreward_libero.py (same rng streams) so
    the episodes also match the standalone 2.0/2.1 runs. Returns per-suite:
      scored: [video_path], donors: [(instruction, video_path)], instr_map.
    """
    rng = random.Random(seed)
    bank_rng = random.Random(seed + 1)
    plan = {}
    for suite in SUITES:
        suite_dir = os.path.join(data_root, suite)
        videos = find_episode_videos(suite_dir, view)
        instr_map = load_episode_instructions(suite_dir)
        by_instr = defaultdict(list)
        for vp in videos:
            by_instr[instr_map.get(episode_index_from_path(vp), "")].append(vp)
        chosen = bank_rng.sample(list(by_instr), min(n_donor, len(by_instr)))
        donors = [(instr, bank_rng.choice(by_instr[instr])) for instr in chosen]
        scored = rng.sample(videos, min(n_episodes, len(videos)))
        plan[suite] = {"scored": scored, "donors": donors, "instr_map": instr_map}
    return plan


# ----------------------------- language lookup ------------------------------ #
class T5LangLookup:
    """Map a raw task string -> cached T5 (context, mask) predictor_kwargs."""

    def __init__(self, cache_dir, context_len, device, dtype):
        self.cache_dir = cache_dir
        self.context_len = context_len
        self.device = device
        self.dtype = dtype

    def __call__(self, instruction):
        prompt = DEFAULT_PROMPT.format(task=instruction)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = os.path.join(self.cache_dir, f"{digest}.t5_len{self.context_len}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing T5 cache for instruction: {path}")
        payload = torch.load(path, map_location="cpu")
        lang = payload["context"].unsqueeze(0).to(self.device, self.dtype)   # [1, L, D]
        lang_mask = payload["mask"].unsqueeze(0).to(self.device)             # [1, L]
        return {"lang": lang, "lang_mask": lang_mask}


# ------------------------------ model runners ------------------------------- #
def make_run_2p0(encoder, target, predictor, img_size, lang_lookup, common):
    """Return run(video_tensor, donor_list, instr) -> dict for a 2.0-style model."""
    def run(vt, donor_list, instr):
        pk = lang_lookup(instr) if lang_lookup is not None else None
        true_v, persist_v, shuffle_v, donor_out = compute_vjepa_loss_sliding_window(
            video_tensor=vt, encoder=encoder, target_encoder=target, predictor=predictor,
            img_size=img_size, masking_mode="causal", is_vae_output=True, mode="mean",
            return_baselines=True, donor_list=donor_list, predictor_kwargs=pk, **common,
        )
        return {"true": true_v, "persist": persist_v, "shuffle": shuffle_v, "donor_out": donor_out}
    return run


def make_run_2p1(encoder, target, predictor, embed_dim, levels, img_size, common):
    """Return run(...) for V-JEPA 2.1; report the last-layer 1408-d slice."""
    def run(vt, donor_list, instr):
        r = compute_vjepa21_loss_sliding_window(
            vt, encoder, target, predictor, embed_dim=embed_dim, levels=levels,
            img_size=img_size, is_vae_output=True, mode="mean", donor_list=donor_list,
            **common,
        )
        return {"true": r["true_last"], "persist": r["persist_last"],
                "shuffle": r["shuffle_last"], "donor_out": r["donor_out"]}
    return run


# --------------------------------- driver ----------------------------------- #
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def evaluate(name, plan, run, load_video, view, min_frames=1):
    """Score one model over the shared plan; return (overall_means, per_suite).

    ``min_frames`` (== window_size) drops any episode whose fps-subsampled clip
    is shorter than one sliding window; the criterion is video-length only, so
    the SAME episodes are dropped for every model and the comparison stays fair.
    """
    print(f"\n{'='*66}\nEvaluating: {name}\n{'='*66}")

    def _load_ok(path):
        vt = load_video(path)
        return vt if vt.shape[2] >= min_frames else None

    # donor bank (GT future latents for the shuffle baseline), in this model's space
    bank = {s: [] for s in SUITES}
    for suite in SUITES:
        for instr, path in plan[suite]["donors"]:
            vt = _load_ok(path)
            if vt is None:
                continue
            d = run(vt, None, instr)["donor_out"]
            if d is not None:
                bank[suite].append((d, instr))

    agg = {"true": [], "persist": [], "shuffle": []}
    per_suite = {}
    skipped_total = 0
    for suite in SUITES:
        st = {"true": [], "persist": [], "shuffle": []}
        skipped = 0
        instr_map = plan[suite]["instr_map"]
        for path in plan[suite]["scored"]:
            vt = _load_ok(path)
            if vt is None:
                skipped += 1
                continue
            instr = instr_map.get(episode_index_from_path(path), "")
            # shuffle donors = same-suite, DIFFERENT instruction
            donors = [d for d, di in bank[suite] if di != instr]
            r = run(vt, donors, instr)
            for k in ("true", "persist", "shuffle"):
                if r[k] is not None:
                    st[k].append(r[k])
                    agg[k].append(r[k])
        skipped_total += skipped
        per_suite[suite] = {k: _mean(v) for k, v in st.items()}
        print(f"  {suite}: true={_mean(st['true']):.4f} "
              f"persist={_mean(st['persist']):.4f} shuffle={_mean(st['shuffle']):.4f} "
              f"(n={len(st['true'])}, skipped {skipped} short)")
    if skipped_total:
        print(f"  [note] skipped {skipped_total} episodes shorter than window_size={min_frames}")
    return {k: _mean(v) for k, v in agg.items()}, per_suite


def main():
    p = argparse.ArgumentParser(
        description="Compare V-JEPA2 / V-JEPA2.1 / lang-conditioned predictor on LIBERO"
    )
    p.add_argument("--data_root", type=str, default=DATA_ROOT)
    p.add_argument("--view", type=str, default="observation.images.image")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--max_frames", type=int, default=49)
    p.add_argument("--target_fps", type=int, default=4)
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--context_frames", type=int, default=2)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_donor", type=int, default=3)
    # our language-conditioned predictor
    p.add_argument("--lang_ckpt", type=str,
                   default="runs/langwam_vitg_libero4_wm_t5xxl/checkpoint_step008460.pt")
    p.add_argument("--text_cache_dir", type=str, default="./data/text_embeds/libero_t5xxl")
    p.add_argument("--lang_dim", type=int, default=4096)
    p.add_argument("--context_len", type=int, default=32)
    p.add_argument("--pred_embed_dim", type=int, default=384)
    p.add_argument("--pred_depth", type=int, default=12)
    p.add_argument("--pred_num_heads", type=int, default=12)
    p.add_argument("--num_mask_tokens", type=int, default=10)
    p.add_argument("--skip_v21", action="store_true", help="Skip the V-JEPA 2.1 model")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan = build_eval_plan(args.data_root, args.view, args.n_episodes, args.n_donor, args.seed)
    n_scored = sum(len(plan[s]["scored"]) for s in SUITES)
    print(f"Shared eval plan: {n_scored} episodes across {len(SUITES)} suites "
          f"(seed={args.seed}, n_donor={args.n_donor})")

    # 2.0-family common kwargs (V-JEPA2 baseline and Ours use these identical settings)
    common_2p0 = dict(window_size=args.window_size, loss_exp=2,
                      context_frames=args.context_frames, seed=args.seed, stride=args.stride)
    common_2p1 = dict(window_size=args.window_size, context_frames=args.context_frames,
                      stride=args.stride)

    def load_video_256(path):
        return load_video_as_tensor(path, max_frames=args.max_frames, img_size=256,
                                    target_fps=args.target_fps).to(device)

    results = {}

    # ---- Model 1 + 3 share ONE frozen encoder, built as in training (uniform_power=False) ----
    print("\nLoading V-JEPA 2 ViT-giant encoder (uniform_power=False, matching training + hub)...")
    _, _, base_predictor, img256 = load_vjepa_models("vitg")  # hub pretrained predictor only
    base_predictor = base_predictor.to(device).eval()
    encoder = build_training_encoder(device, img_size=256, num_frames=args.window_size)
    target_encoder = encoder  # frozen and identical -> reuse for GT latents
    model_dtype = next(encoder.parameters()).dtype

    run_v2 = make_run_2p0(encoder, target_encoder, base_predictor, img256, None, common_2p0)
    results["V-JEPA2 (vitg,256)"] = evaluate(
        "V-JEPA2 (vitg, 256) - pretrained predictor", plan, run_v2, load_video_256, args.view,
        min_frames=args.window_size)

    # ---- Model 3: ours (same encoder + our lang predictor) ----
    print(f"\nLoading our language-conditioned predictor: {args.lang_ckpt}")
    lang_predictor = lang_vit_predictor(
        img_size=256, patch_size=encoder.patch_size, num_frames=args.window_size,
        tubelet_size=encoder.tubelet_size, embed_dim=encoder.embed_dim,
        predictor_embed_dim=args.pred_embed_dim, depth=args.pred_depth,
        num_heads=args.pred_num_heads, uniform_power=False, use_mask_tokens=True,
        num_mask_tokens=args.num_mask_tokens, zero_init_mask_tokens=True, use_rope=True,
        lang_dim=args.lang_dim,
    )
    ck = torch.load(args.lang_ckpt, map_location="cpu", weights_only=False)
    print("  predictor load:", lang_predictor.load_state_dict(ck["predictor"]))
    lang_predictor = lang_predictor.to(device).eval()
    lang_lookup = T5LangLookup(args.text_cache_dir, args.context_len, device, model_dtype)
    run_ours = make_run_2p0(encoder, target_encoder, lang_predictor, img256, lang_lookup, common_2p0)
    results["Ours (lang-vitg,256)"] = evaluate(
        "Ours (frozen vitg encoder + T5-XXL FiLM predictor)", plan, run_ours,
        load_video_256, args.view, min_frames=args.window_size)

    # ---- Model 2: V-JEPA 2.1 giant-384 ----
    if not args.skip_v21:
        print("\nLoading V-JEPA 2.1 ViT-giant-384...")
        e21, t21, p21, img384, embed_dim, levels = load_vjepa21_giant()
        e21 = e21.to(device).eval(); t21 = t21.to(device).eval(); p21 = p21.to(device).eval()

        def load_video_384(path):
            return load_video_as_tensor(path, max_frames=args.max_frames, img_size=384,
                                        target_fps=args.target_fps).to(device)

        run_v21 = make_run_2p1(e21, t21, p21, embed_dim, levels, img384, common_2p1)
        results["V-JEPA2.1 (vitg,384)"] = evaluate(
            "V-JEPA2.1 (vitg-384) - last-1408d slice", plan, run_v21, load_video_384, args.view,
            min_frames=args.window_size)

    # ------------------------------ summary table --------------------------- #
    print(f"\n{'='*90}")
    print("SUMMARY  -  similarity = 1 - surprise (higher = better).  "
          "skill = 1 - true_surprise/floor_surprise")
    print(f"{'-'*90}")
    print(f"{'model':<26}{'sim_true':>10}{'sim_persist':>13}{'sim_shuffle':>13}"
          f"{'skill_vs_p':>12}{'skill_vs_s':>12}")
    for name, (m, _) in results.items():
        t, pv, s = m["true"], m["persist"], m["shuffle"]
        print(f"{name:<26}{1-t:>10.4f}{1-pv:>13.4f}{1-s:>13.4f}"
              f"{1-t/pv:>12.4f}{1-t/s:>12.4f}")
    print(f"{'='*90}")
    print(f"({n_scored} episodes; window={args.window_size} context={args.context_frames} "
          f"stride={args.stride}; surprise space = last-layer 1408-d cosine)")


if __name__ == "__main__":
    main()
