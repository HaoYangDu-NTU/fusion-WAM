"""Unified single-bucket LeRobot v3 reader base.

One base class for every LeRobot v3 single-bucket reader in this package —
the 4 OXE readers (BC-Z / Bridge / Fractal / DROID), RoboCOIN, and EgoDex are
all **direct, equal subclasses** of :class:`LeRobotV3Reader`. There is no
per-family intermediate base; the differences between readers are expressed
through a small set of hook methods + class attributes, and the genuinely
shared building blocks (single-arm EEF assembly, stats materialization,
prompt resolution, multi-bucket construction) live as stateless helpers in
``openwam.dataloader.utils``.

The canonical sample dict every subclass emits::

    {
      "video": [PIL.Image x num_video_frames],
      "vace_video": None,
      "clean_condition_images": [video[0]],
      "action": (T-1, action_dim) float32,
      "action_mask": (T-1, action_dim) bool,
      "video_mask": (num_video_frames,) bool,
      "proprio": (1, action_dim) float32,
      "proprio_mask": (1, action_dim) bool,
      "prompt": str,
    }

With ``history_num_tubelets=H>0``, ``clean_condition_images`` contains 2H
frames at stride 4 ending at the current observation; past indices clamp at
episode frame 0. H=1 selects ``[episode[max(t-4, 0)], video[0]]``. Video/action windows and masks still
start at the current frame; history shares the same view selection and jitter.

Subclass surface
----------------
Class attributes:
  * ``DATASET_NAME``        — log / error label.
  * ``NEEDED_COLS``         — parquet columns the reader consumes.
  * ``HEAD_CAMERA`` / ``LEFT_WRIST_CAMERA`` / ``RIGHT_WRIST_CAMERA`` — used by
    the default :meth:`_resolve_cameras`.
  * ``ACTION_DIM_MASK``     — per-dim validity mask (``LEFT_ARM_DIM_MASK`` for
    single-arm OXE; ``None`` = all dims valid, for bimanual RoboCOIN).
  * ``PROPRIO_DIM_MASK``    — optional state-specific validity mask; ``None``
    inherits ``ACTION_DIM_MASK`` for backward compatibility.
  * ``ACTION_DIM``          — per-step action/proprio width (20 = EEF).
  * ``PROMPT_FILE_REQUIRED``— default :meth:`_load_prompts` behavior on a
    missing ``tasks.parquet`` (True = raise, False = empty map).

Hooks (override as needed; defaults cover the common case):
  * ``_resolve_cameras(info)``     — (head, left_wrist, right_wrist).
  * ``_add_data_offsets(eps)``     — set ``eps['_data_row_offset']``.
  * ``_train_min_window_len()``    — min episode length for a train window.
  * ``_load_prompts()``            — populate the prompt lookup.
  * ``_resolve_prompt(row, win)``  — per-sample prompt text.
  * ``_load_stats(info)``          — return in-reader normalization stats / None.
  * ``_action_20d(win)`` / ``_proprio_20d(win)`` — normalized (..., 20) payload,
    or ``None`` to emit zeros (video-only / disabled supervision).
  * ``_post_init(info)``           — one-time per-dataset setup.
  * ``_multibucket_wrapper()``     — root-mode aggregate wrapper class or None.
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
import random
import socket
import uuid
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from openwam.history import condition_frame_offsets, history_from_config, validate_history_num_tubelets, validate_condition_frame_offsets
from openwam.dataloader.bases.dataset import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout, assemble_two_view_layout
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.eef import (
    EEF_DIM,
    build_action_mask_2d,
    build_proprio_mask_2d,
)
from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits,
    build_multibucket,
    compute_file_local_offsets,
    effective_episode_frames,
    load_episodes_parquet,
    load_tasks_annotated,
    parse_info_json,
    resolve_prompt_by_episode,
    subsample_episodes_by_hours,
)
from openwam.dataloader.utils.normalization import materialize_eef_stats
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec
from openwam.dataloader.utils.video_io import decode_video_frames as _decode_video_frames

logger = logging.getLogger(__name__)

# Multiview L-shape slot sizes (must match assemble_multiview_layout defaults).
_HEAD_SLOT_H, _HEAD_SLOT_W = 256, 320
_WRIST_SLOT_H, _WRIST_SLOT_W = 128, 160
_GETITEM_MAX_RETRIES = 64

# Sentinel so an explicit ``normalize_mode=None`` (disable normalization) is
# distinguishable from "not passed" (fall back to the subclass default).
_NORMALIZE_MODE_UNSET = object()

# Sentinel for from_config key-presence checks: distinguishes a config key that
# is absent from one explicitly set to None.
_CONFIG_MISSING = object()


# Process-global LRU for decoded parquet shards, shared across all bucket
# instances. A per-instance cache (the previous maxsize=32 design) is a memory
# risk under shuffled training: bucket count, shard count, and DataLoader
# workers multiply the number of retained tables while random access provides
# little reuse. A small global cache keeps sequential scans fast with bounded
# per-worker memory.
@functools.lru_cache(maxsize=4)
def _read_data_table_cached(path: str, columns: Tuple[str, ...]):
    try:
        return pq.read_table(path, memory_map=True, columns=list(columns))
    except pa.ArrowInvalid as exc:
        # Some pyarrow versions interpret flat LeRobot feature names containing
        # dots as nested-field paths. Keep the compatibility fallback inside
        # the process-global cache so affected shards are still read only once.
        if "Dot path" not in str(exc):
            raise
        return pq.read_table(path, memory_map=True)


class LeRobotV3Reader(BaseDataset):
    """Shared single-bucket LeRobot v3 reading machinery (see module docstring)."""

    # --- subclass-overridable class attributes ---
    DATASET_NAME: ClassVar[str] = "LeRobotV3"
    NEEDED_COLS: ClassVar[Tuple[str, ...]] = ()
    HEAD_CAMERA: ClassVar[Optional[str]] = None
    LEFT_WRIST_CAMERA: ClassVar[Optional[str]] = None
    RIGHT_WRIST_CAMERA: ClassVar[Optional[str]] = None
    ACTION_DIM_MASK: ClassVar[Optional[np.ndarray]] = None
    # Optional proprio-specific raw-dimension mask.  ``None`` preserves the
    # historical contract by inheriting ``ACTION_DIM_MASK``; readers whose
    # action/state schemas differ may set an instance value in
    # ``_resolve_cameras``.
    PROPRIO_DIM_MASK: ClassVar[Optional[np.ndarray]] = None
    ACTION_DIM: ClassVar[int] = EEF_DIM
    # Prompt convention: "task_index" (tasks.parquet, index=text — RoboCOIN /
    # EgoDex) or "episode_annotated" (tasks_annotated.parquet by episode_index — OXE).
    PROMPT_SOURCE: ClassVar[str] = "task_index"
    PROMPT_FILE_REQUIRED: ClassVar[bool] = True
    # In-reader normalization stats: when STATS_FILENAME is set, the default
    # ``_load_stats`` loads ``meta/<STATS_FILENAME>`` (a flat stats dict) with
    # the given dim / min-max strictness (OXE single-arm = 10-D, strict).
    # RoboCOIN overrides ``_load_stats`` for its dynamic per-robot-type path.
    STATS_FILENAME: ClassVar[Optional[str]] = None
    STATS_DIM: ClassVar[int] = EEF_DIM
    STATS_STRICT_MINMAX: ClassVar[bool] = False
    # Deploy denormalizer artifact (meta/normalization_stats.npy). A reader that
    # serves a unified action and wants a deployable checkpoint sets this to the
    # action_mode key its RAW stats are stored under, and calls
    # ``_write_deploy_normalizer_stats(combined, keys)`` from its ``_load_stats``.
    # At deploy the policy server's ``_UnifyAwareNormalizer`` gathers the model's
    # unified output back to raw dims, THEN unnormalizes with these RAW stats —
    # so the artifact is authored in RAW space (NOT scattered). None → no deploy
    # stats written (default).
    DEPLOY_ACTION_MODE: ClassVar[Optional[str]] = None
    # Default normalize_mode when the caller doesn't pass one. OXE readers
    # override to "quantile" (their historical default); RoboCOIN/EgoDex keep
    # None (no in-reader normalization unless a config opts in).
    DEFAULT_NORMALIZE_MODE: ClassVar[Optional[str]] = None
    # Exception types a wrist (auxiliary-camera) decode failure is tolerated for
    # (missing / corrupt mp4 → black slot rather than a fatal sample). OXE used
    # the specific triple; RoboCOIN tolerated all. Subclasses override as needed.
    WRIST_DECODE_TOLERATED: ClassVar[tuple] = (FileNotFoundError, OSError, RuntimeError)

    def __init__(
        self,
        dataset_dir: str,
        *,
        num_frames: int = 33,
        video_stride: int = 4,
        history_num_tubelets: int = 0,
        clean_condition_frame_offsets=None,
        seperate_encode_view: bool = False,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        split: str = "train",       # LINK: openwam/dataloader/bases/lerobot_v3_reader.py:582
        multiview: bool = False,
        normalize_mode: Any = _NORMALIZE_MODE_UNSET,
        dataset_id: Optional[str] = None,
        target_camera: Optional[str] = None,
        camera_layout: Optional[List[str]] = None,
        # Octo-style head-view sampling: a list of >=2 camera keys (must include
        # the resolved head camera). Each TRAIN window decodes the head slot from
        # ONE uniformly sampled entry — viewpoint augmentation at unchanged window
        # count. Val / None / single entry → always the resolved head camera.
        head_camera_choices: Optional[List[str]] = None,
        # Unified action space. unify_action=True scatters this reader's raw
        # ACTION_DIM-wide action/proprio into a UNIFY_DIM-wide vector and marks
        # only the mapped dims valid. unify_action_map is the yaml spec (see
        # openwam.dataloader.utils.unify_action); None + unify_action=True maps
        # the raw dims in order (0..raw_dim-1). The unified width is the global
        # UNIFY_DIM constant (not per-dataset configurable).
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        unify_state_map: Optional[Any] = None,
        # Optional load-time video color jitter, applied consistently across a
        # clip's frames and ONLY on the train split. None / False / {} → disabled
        # (default). Truthy → enabled; a dict overrides
        # the per-channel strengths {brightness, contrast, saturation, hue}.
        color_jitter: Optional[Any] = None,
        # Optional data-budget knobs (None = use the full bucket).
        max_hours: Optional[float] = None,      # LINK: openwam/dataloader/bases/lerobot_v3_reader.py:390
        subsample_seed: int = 42,
        **_unused: Any,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._dataset_id = dataset_id or self._dataset_dir.name
        self._num_frames = int(num_frames)
        self._video_stride = max(1, int(video_stride))
        self._history_num_tubelets = validate_history_num_tubelets(history_num_tubelets)
        self._clean_condition_frame_offsets = (
            validate_condition_frame_offsets(clean_condition_frame_offsets)
            if clean_condition_frame_offsets is not None else None
        )
        if self._clean_condition_frame_offsets is not None and self._history_num_tubelets:
            raise ValueError("Use explicit clean frame offsets or history_num_tubelets, not both")
        self._window_stride = max(1, int(window_stride))
        self._seperate_encode_view = bool(seperate_encode_view)
        self._height = 224 if self._seperate_encode_view else int(height)
        self._width = 448 if self._seperate_encode_view else int(width)
        self._split = split
        self._multiview = bool(multiview)
        self._normalize_mode = (
            self.DEFAULT_NORMALIZE_MODE if normalize_mode is _NORMALIZE_MODE_UNSET else normalize_mode
        )
        self._target_camera = target_camera
        self._camera_layout_param = list(camera_layout) if camera_layout else None
        self._max_hours = max_hours
        self._subsample_seed = int(subsample_seed)

        # ── load-time video augmentation ──────────────────────────────────
        # Color jitter is applied in _getitem_impl to the decoded clip (same
        # random factors across all frames, via VideoColorJitter). Built only
        # for the train split; val / disabled keeps video byte-identical.
        self._color_jitter = None
        if color_jitter_enabled(color_jitter) and split == "train":
            cj_get = color_jitter.get if hasattr(color_jitter, "get") else (lambda k, d: d)
            self._color_jitter = VideoColorJitter(
                brightness=float(cj_get("brightness", 0.2)),
                contrast=float(cj_get("contrast", 0.2)),
                saturation=float(cj_get("saturation", 0.2)),
                hue=float(cj_get("hue", 0.0)),
            )

        # ── unified action space ──────────────────────────────────────────
        # _raw_action_dim is what this reader's _action_20d/_proprio_20d emit
        # (the class ACTION_DIM). When unify is on, the public ACTION_DIM (and
        # thus the finalized action/proprio width + downstream model action_dim)
        # becomes unify_dim, and _finalize_* scatters raw -> unified via
        # _unify_dst_index. Off (default) → raw layout kept.
        # Instance attr (not type(self).ACTION_DIM) so a subclass can override the
        # raw action width per bucket BEFORE super().__init__ — RoboCOIN sets a
        # wider raw dim (pose + dexterous-hand fingers) for dex-hand buckets under
        # unify. Defaults to the class ACTION_DIM, so existing readers are unchanged.
        self._raw_action_dim = int(self.ACTION_DIM)
        self._unify = bool(unify_action)
        self._unify_dim = int(UNIFY_DIM)

        # 根据 unify_action_map 获取具体在 80D space 中的位置 （action 和 state 都处理）
        self._unify_dst_index: Optional[np.ndarray] = None
        self._unify_state_dst_index: Optional[np.ndarray] = None
        if self._unify:
            if unify_action_map is None:
                # No spec → identity map: raw dims 0..raw_dim-1 in order.
                # Single-list form (top-level ints), NOT [[...]] which the
                # parser would read as one malformed src->dst pair.
                spec = list(range(self._raw_action_dim))
            else:
                spec = unify_action_map
            self._unify_dst_index = parse_unify_spec(spec, self._unify_dim)
            state_spec = unify_state_map if unify_state_map is not None else spec
            self._unify_state_dst_index = parse_unify_spec(state_spec, self._unify_dim)
            if self._unify_state_dst_index.shape[0] != self._raw_action_dim:
                raise ValueError(
                    f"{self.DATASET_NAME} unify_state_map maps "
                    f"{self._unify_state_dst_index.shape[0]} source dims but this reader emits "
                    f"{self._raw_action_dim}-D state"
                )
            if self._unify_dst_index.shape[0] != self._raw_action_dim:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): unify_action_map maps "
                    f"{self._unify_dst_index.shape[0]} source dims but this reader emits "
                    f"{self._raw_action_dim}-D action. They must match."
                )
            # Instance attr shadows the class ACTION_DIM so finalized payloads,
            # the action_dim property, and the model action head are all unify_dim.
            self.ACTION_DIM = self._unify_dim

        # Deploy denormalizer artifact path (set by _write_deploy_normalizer_stats
        # when a reader emits meta/normalization_stats.npy; None otherwise).
        self.normalization_stats_path: Optional[str] = None

        # Rate-limited failure counters for _safe_get / wrist decode.
        self._fail_count = 0
        self._wrist_fail_count = 0
        self._fail_log_every = 100

        # ── info.json ─────────────────────────────────────────────────────
        info = parse_info_json(self._dataset_dir)
        self._fps = float(info["fps"])
        self._data_path_template = info["data_path"]
        self._video_path_template = info["video_path"]

        # ── cameras + multiview layout ────────────────────────────────────
        self._head_camera, self._left_wrist_camera, self._right_wrist_camera = self._resolve_cameras(info)
        if self._head_camera is None:
            raise ValueError(f"{self.DATASET_NAME}({self._dataset_id}): no head camera resolved")
        # Octo-style head-view sampling (see __init__ arg docs). Resolved BEFORE
        # _build_episode_index so _video_cameras() covers every choice when the
        # per-camera frame offsets are computed. Randomness comes from the
        # module-level ``random`` RNG — per-worker seeded by the torch worker
        # loop (and by dataloader_worker_init_fn in deterministic mode), the
        # same source the video transforms use.
        # Val keeps the resolved head camera so eval stays deterministic.
        self._head_camera_choices: Optional[List[str]] = None
        if head_camera_choices:
            cams = list(dict.fromkeys(str(c) for c in head_camera_choices))
            if self._head_camera not in cams:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): head_camera_choices {cams} must include "
                    f"the resolved head camera {self._head_camera!r} (the val/default view)."
                )
            if self._split == "train" and len(cams) > 1:
                self._head_camera_choices = cams
        if self._multiview:
            self._camera_layout = self._camera_layout_param or [
                self._head_camera,
                self._left_wrist_camera or "__missing_left__",
                self._right_wrist_camera or "__missing_right__",
            ]
        else:
            self._camera_layout = [self._head_camera]

        if self._seperate_encode_view and (
            not self._multiview or self._left_wrist_camera is None or self._right_wrist_camera is not None
        ):
            raise ValueError("seperate_encode_view requires exactly two configured cameras: head and left wrist")

        # ── video sub-sampling ────────────────────────────────────────────
        # video_stride takes every Nth frame within the window. num_video_frames
        # is whatever that yields; for clean encoder temporal downsampling it
        # should match the encoder's contract (Wan VAE: (num_video_frames - 1) % 4
        # == 0) — NOT enforced here, a mismatch surfaces downstream at encode time.
        self._video_sample_indices = np.arange(0, self._num_frames, self._video_stride, dtype=np.int64)
        self._num_video_frames = int(self._video_sample_indices.size)

        # ── episodes + offsets + split (hook) ─────────────────────────────
        # _build_episode_index is overridable so non-v3 on-disk layouts (e.g.
        # LeRobot v2.1: per-episode parquet + meta/episodes.jsonl) can supply
        # the same eps DataFrame contract without reimplementing __init__.

        # 读取 lerobot原始 meta/episodes 下的 parquet + 预计算三个偏移值
        self._eps_df = self._build_episode_index(info)

        # Stable universe for external scan tools' over-exclusion guardrails. Capture
        # it before this reader applies the mutable exclusion artifact; using a
        # post-exclusion length plus a later artifact snapshot races concurrent
        # exclusion writers and can overstate the denominator.
        self._n_episodes_before_exclusions = int(len(self._eps_df))

        # ── per-bucket episode exclusion (data-quality blacklist) ─────────
        # meta/excluded_episodes.json holds episode_index values that must not
        # be sampled (e.g. episodes whose frames fall in a truncated video
        # file). Rows are dropped AFTER offset computation — the same
        # alignment-safe filter path as info splits; physically deleting
        # episodes-parquet rows would shift the groupby-cumsum offsets of
        # later episodes in each (chunk, file) shard and misalign them.
        # NOTE: 所以这里可以维护一个 excluded_episodes.json 来排除不要的 episode
        # 这里是先算偏移，再排除：因为被排除的 episode 仍然占据磁盘文件中的数据行和视频帧，这里不会删除实际文件内容。
        excluded = self._load_excluded_episode_indices()
        if excluded:
            n_before = len(self._eps_df)
            self._eps_df = self._eps_df[~self._eps_df["episode_index"].isin(excluded)].reset_index(drop=True)
            if len(self._eps_df) < n_before:
                logger.info(
                    "%s(%s): excluded %d/%d episodes via meta/excluded_episodes.json",
                    self.DATASET_NAME,
                    self._dataset_id,
                    n_before - len(self._eps_df),
                    n_before,
                )

        # ── subclass episode filter hook (default identity) ───────────────
        # Runs AFTER info-split + excluded_episodes.json filtering and BEFORE
        # the offset arrays are extracted, so dropped rows stay alignment-safe
        # (the per-row _data_row_offset / _video_frame_offset columns ride
        # along). Ego4D overrides this to drop episodes whose bilingual prompt
        # has no usable English half; every other reader keeps the identity
        # default.

        # NOTE: 这里在后续 agibotworld / robocoin / interndata_a1 中重写，用来筛掉 episode 内开始/结尾的一些帧  ----> 这个写法可以参考
        # 本质上就是重新确定 episode 的起止点，修改可见范围，而不用直接修改原始数据
        self._eps_df = self._filter_episodes(self._eps_df)

        # ── optional episode-level subsample to fit a per-bucket hour budget ──
        if self._max_hours is not None:
            self._select_episodes_by_hours(float(self._max_hours), self._subsample_seed)


        # NOTE: 走完前面两步过滤后得到的最终_eps_df，通过这个函数生成后续 getitem 时要使用的索引数组，并计算每条 episode 能产生多少个训练窗口, 定义 getitem 时 idx 的合法范围
        self._rebuild_episode_arrays()
        if self._head_camera_choices:
            # Fail fast on a mistyped choice: a camera without video index
            # columns would otherwise surface as an opaque empty-head-decode
            # retry storm in _safe_get.
            missing = [c for c in self._head_camera_choices if c not in self._ep_video_frame_offsets]
            if missing:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): head_camera_choices cameras {missing} "
                    "have no videos/<cam>/chunk_index + file_index columns in the episodes table."
                )
            logger.info(
                "%s(%s): head-view sampling active over %s",
                self.DATASET_NAME,
                self._dataset_id,
                self._head_camera_choices,
            )

        # ── prompts + per-dataset normalization stats (hooks) ─────────────
        self._load_prompts()
        self._normalization_stats = self._load_stats(info)

        # ── subclass hook ─────────────────────────────────────────────────
        self._post_init(info)

        # ── unified per-dim validity mask (static) ────────────────────────
        # Honors the raw ACTION_DIM_MASK: single-arm OXE zero-pads
        # the right-arm half of the 20-D EEF, which must stay masked out even
        # after the scatter — a unified slot is valid iff its source raw dim
        # was valid. Built HERE (not in the unify block above) because a
        # reader's instance ACTION_DIM_MASK may be set in _resolve_cameras
        # (some readers set it per-embodiment), which runs after that block.
        self._unify_dim_mask: Optional[np.ndarray] = None
        self._unify_proprio_dim_mask: Optional[np.ndarray] = None
        if self._unify:
            self._unify_dim_mask = np.zeros(self._unify_dim, dtype=bool)
            action_raw_mask = self.ACTION_DIM_MASK
            if action_raw_mask is None:
                self._unify_dim_mask[self._unify_dst_index] = True
            else:
                self._unify_dim_mask[self._unify_dst_index] = np.asarray(action_raw_mask, dtype=bool)

            # Most LeRobot sources expose action and proprio in the same raw
            # schema, so an unset proprio mask inherits the action mask.  A
            # source may override this when a field exists only on one stream
            # (for example, a commanded mobile-base velocity with no measured
            # state velocity).
            proprio_raw_mask = self.PROPRIO_DIM_MASK
            if proprio_raw_mask is None:
                proprio_raw_mask = action_raw_mask
            self._unify_proprio_dim_mask = np.zeros(self._unify_dim, dtype=bool)
            if proprio_raw_mask is None:
                self._unify_proprio_dim_mask[self._unify_state_dst_index] = True
            else:
                self._unify_proprio_dim_mask[self._unify_state_dst_index] = np.asarray(proprio_raw_mask, dtype=bool)

        logger.info(
            "%s(%s, %s): %d eps, %d windows, fps=%.1f, multiview=%s, normalize=%s",
            self.DATASET_NAME,
            self._dataset_id,
            split,
            len(self._eps_df),
            self._n_total,
            self._fps,
            self._multiview,
            self._normalize_mode,
        )

    def _select_episodes_by_hours(self, target_hours: float, seed: int) -> None:
        """Select whole episodes against a post-filter, effective-hour budget.

        This helper only changes ``_eps_df``.  The caller must rebuild the
        derived episode/window arrays afterwards.  Keeping selection separate
        lets root-mode multi-bucket construction first build every leaf against
        its complete split/exclusion/trim population, water-fill using those
        *effective* capacities, and then apply each allocation in place.
        """
        n_before = len(self._eps_df)
        if n_before == 0:
            raise ValueError(
                f"{self.DATASET_NAME}({self._dataset_id}): eps_df is already empty before "
                "subsample (likely split/info.json mismatch)."
            )
        self._max_hours = float(target_hours)
        self._subsample_seed = int(seed)
        self._eps_df = subsample_episodes_by_hours(
            self._eps_df,
            target_hours=self._max_hours,
            fps=self._fps,
            seed=self._subsample_seed,
            episode_frames=self._sampleable_episode_frames(self._eps_df),
        )
        if len(self._eps_df) == 0:
            raise ValueError(
                f"{self.DATASET_NAME}({self._dataset_id}): subsampling to max_hours={self._max_hours}h left 0 episodes."
            )
        logger.info(
            "%s(%s): subsampled %d/%d episodes (max_hours=%.3f effective h, seed=%d)",
            self.DATASET_NAME,
            self._dataset_id,
            len(self._eps_df),
            n_before,
            self._max_hours,
            self._subsample_seed,
        )

    def _rebuild_episode_arrays(self) -> None:
        """Rebuild every array derived from the current ``_eps_df`` rows.
        这些数组的相同位置对应同一条 episode: 
            self._eps_df.iloc[i]
            self._ep_valid_start[i]
            self._ep_valid_end[i]
            self._ep_data_row_offset[i]
            self._ep_video_frame_offsets[cam][i]
        """
        effective_length = effective_episode_frames(self._eps_df)
        self._ep_valid_start = (
            self._eps_df["_valid_start"].to_numpy().astype(np.int64)
            if "_valid_start" in self._eps_df.columns
            else np.zeros(len(self._eps_df), dtype=np.int64)
        )
        self._ep_valid_end = (
            self._eps_df["_valid_end"].to_numpy().astype(np.int64)
            if "_valid_end" in self._eps_df.columns
            else self._eps_df["length"].to_numpy().astype(np.int64)
        )
        self._ep_data_row_offset = self._eps_df["_data_row_offset"].to_numpy().astype(np.int64)
        self._ep_video_frame_offsets = {}
        for cam in self._video_cameras():
            col = self._video_offset_col(cam)
            if col in self._eps_df.columns:
                self._ep_video_frame_offsets[cam] = self._eps_df[col].to_numpy().astype(np.int64)

        # NOTE: 这里是决定 padding 的核心参数，这个值表示剩余至少多少行即可作为窗口起点 (倒推回去，最大 padding 数就等于 window size 减去这个值)，而默认设为 1
        min_window_len = self._num_frames if self._split == "val" else self._train_min_window_len()

        # n_starts 表示每个 episode 合法起点位置的数量，考虑了上面的 min_window_len，因此这里得到的 n_starts 就是在允许 padding 范围内的起点合集
        n_starts = np.where(
            effective_length >= min_window_len,
            (effective_length - min_window_len) // self._window_stride + 1,
            0,
        ).astype(np.int64)

        # 累加合法起点，就可以得到连续索引，标记每条 episode 的样本编号边界，这个是每条 episode 的训练样本编号分界线
        # 这里面的值和 valid_start / valid_end 不一样，这两个 valid 是原始 episode 内可用时间步的起止点
        # _cum_n_starts 记录的是真实训练时窗口的编号起止点，要在后续 getitem 里找到具体某个窗口 idx 和原始 episode idx 的对应关系
        self._cum_n_starts = np.concatenate([[0], np.cumsum(n_starts)]).astype(np.int64)
        # 最后一个就是合法起点总数，修改了 dataset 的 len 方法，因此这个就是标记数据集大小，从而告诉 DataLoader 的 sampler 可以在这个范围内生成索引
        # LINK: openwam/dataloader/bases/lerobot_v3_reader.py:861
        self._n_total = int(self._cum_n_starts[-1])

    def _sampleable_episode_frames(self, eps_df: pd.DataFrame) -> np.ndarray:
        """Effective frames, with non-window-producing episodes charged as zero."""
        frames = effective_episode_frames(eps_df)
        min_window_len = self._num_frames if self._split == "val" else self._train_min_window_len()
        return np.where(frames >= min_window_len, frames, 0).astype(np.int64)

    def _apply_effective_hour_budget(self, target_hours: float, seed: int) -> None:
        """Apply a root-mode water-fill allocation to an initialized leaf.

        Multi-bucket construction deliberately initializes leaves uncapped so
        their split/exclusion/trim filters and data-contract checks run against
        the complete population.  It then calls this method once with the
        effective-hours allocation.  Prompt tables, normalization stats and
        static action masks do not depend on the selected episode subset; only
        the episode dispatch arrays need rebuilding. This must run before the
        multi-bucket wrapper snapshots child lengths into its cumulative index.
        """
        self._select_episodes_by_hours(target_hours, seed)
        self._rebuild_episode_arrays()

    @property
    def effective_hours(self) -> float:
        """Post-split/exclusion/trim footage represented by this leaf."""
        frames = self._sampleable_episode_frames(self._eps_df)
        return float(frames.sum()) / self._fps / 3600.0

    # ----- hooks (defaults) -------------------------------------------------

    def _resolve_cameras(self, info: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Resolve (head, left_wrist, right_wrist) camera keys.

        Default: a single ``target_camera`` if set (EgoDex-style), else the
        class-level ``HEAD_CAMERA`` / ``LEFT_WRIST_CAMERA`` / ``RIGHT_WRIST_CAMERA``
        (OXE-style). RoboCOIN overrides to resolve by info.features priority.
        """
        if self._target_camera is not None:
            return (self._target_camera, None, None)
        return (self.HEAD_CAMERA, self.LEFT_WRIST_CAMERA, self.RIGHT_WRIST_CAMERA)

    def _add_data_offsets(self, eps: pd.DataFrame) -> None:
        """Set ``eps['_data_row_offset']`` (generic LeRobot v3 groupby-cumsum)."""
        eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        """Load + offset + split the episodes table. LeRobot v3 default.

        Returns the split-filtered episodes DataFrame the rest of ``__init__``
        consumes. It MUST carry: ``length``, ``episode_index``,
        ``data/chunk_index``, ``data/file_index``, ``_data_row_offset``, and for
        every resolved camera ``videos/<cam>/chunk_index`` /
        ``videos/<cam>/file_index`` / ``_video_frame_offset/<cam>``.

        Override for non-v3 on-disk layouts (e.g. LeRobot v2.1: one parquet per
        episode + ``meta/episodes.jsonl`` instead of ``meta/episodes/*.parquet``).
        """
        # 读取 lerobot 数据 meta/episodes 的原始 parquet
        eps = load_episodes_parquet(self._dataset_dir)
        # 新增三列，其中 _data_row_offset 标明每条 episode 在其所属 Parquet 数据文件中的起始行号 -> 这个负责 action/state 的读取
        # 类似的，这里对每个视角新增一个 _video_frame_offset/{camera}，标明每条 episode 在对应视频文件中的起始帧
        self._add_episode_offsets(eps)
        info_splits = info.get("splits", {}) or {}
        # 会根据 info.json中的 "splits" 指定的episode idx 来筛选出当前 ‘self._split‘ 需要的 episode idx
        return apply_info_splits(eps, self._split, info_splits, source_name=f"{self.DATASET_NAME}({self._dataset_id})")

    def _load_excluded_episode_indices(self) -> set[int]:
        """Return the episode blacklist used by the construction-time filter.

        The default reads the generic scanner artifact. Subclasses that already
        validated and cached this artifact while building their episode index
        may return that canonical snapshot, preventing a second read from
        observing a concurrent atomic replacement.
        """
        path = self._dataset_dir / "meta" / "excluded_episodes.json"
        if not path.exists():
            return set()

        import json

        with open(path) as f:
            return set(json.load(f)["episode_indices"])

    def _train_min_window_len(self) -> int:
        """Min episode length to yield a train window. 1 = any single labeled step."""
        return 1

    def _filter_episodes(self, eps_df: pd.DataFrame) -> pd.DataFrame:
        """Optional hook to drop episodes after split / excluded_episodes filtering.

        Default: identity (no filtering). Called in ``__init__`` right after the
        ``meta/excluded_episodes.json`` blacklist is applied and before the
        offset arrays / window index are built, so a subclass can remove
        episodes on a data-quality criterion computed from ``eps_df`` columns
        (e.g. Ego4D drops rows whose bilingual ``tasks`` string has no usable
        English half). Implementations MUST ``reset_index(drop=True)`` on the
        returned frame. The per-row ``_data_row_offset`` / ``_video_frame_offset``
        columns are preserved across row filtering, so alignment stays correct.

        An implementation may additionally return two optional columns to trim
        each episode instead of dropping it whole:

        ``_valid_start`` / ``_valid_end``
            Half-open ``[start, end)`` row range of the episode that may be
            sampled. Absent → ``[0, length)`` (the whole episode).
            ``__init__`` builds the window index from ``end - start`` and
            ``_getitem_impl`` offsets BOTH the parquet slice and the video decode
            by ``start``, so the two stay aligned and no window can reach a
            trimmed row or frame. AgiBotWorld uses this for the segment-boundary
            cleanup (see :meth:`~openwam.dataloader.agibotworld.AgiBotWorldDataset._filter_episodes`).

        Contract: emit them as a PAIR, with ``0 <= start < end <= length``. They
        are consumed as given — an out-of-range value would silently read into
        the neighbouring episode, since LeRobot v3 packs many episodes per
        parquet shard and per mp4. Drop the episode rather than returning an
        empty or inverted range.
        """
        return eps_df

    def _load_prompts(self) -> None:
        """Populate the prompt lookup according to ``PROMPT_SOURCE``."""
        if self.PROMPT_SOURCE == "episode_annotated":
            self._episode_idx_to_text = load_tasks_annotated(
                self._dataset_dir, source_name=f"{self.DATASET_NAME}({self._dataset_id})"
            )
            return
        # "task_index": meta/tasks.parquet, index=text, "task_index" column=int.
        tasks_path = self._dataset_dir / "meta" / "tasks.parquet"
        if not self.PROMPT_FILE_REQUIRED and not tasks_path.exists():
            self._task_idx_to_text: Dict[int, str] = {}
            return
        tasks_df = pd.read_parquet(tasks_path)  # FileNotFoundError if required + missing
        task_idx = tasks_df["task_index"].to_numpy()
        task_str = tasks_df.index.to_numpy()
        self._task_idx_to_text = dict(zip(task_idx.tolist(), task_str.tolist()))

    def _resolve_prompt(self, row, win: pd.DataFrame) -> str:
        """Resolve this window's prompt text according to ``PROMPT_SOURCE``.

        Raises on a missing entry / empty string (a data bug should not slip
        into training as a fabricated prompt).
        """
        if self.PROMPT_SOURCE == "episode_annotated":
            return resolve_prompt_by_episode(self._episode_idx_to_text, int(row["episode_index"]), self.DATASET_NAME)
        task_idx = int(win["task_index"].iloc[0])
        if task_idx not in self._task_idx_to_text:
            raise KeyError(
                f"{self.DATASET_NAME} prompt lookup failed: task_index={task_idx} not present "
                "in this bucket's tasks.parquet."
            )
        text = self._task_idx_to_text[task_idx].strip()
        if not text:
            raise ValueError(
                f"{self.DATASET_NAME} task_index={task_idx} maps to an empty prompt string. "
                "Indicates a tasks.parquet entry with blank text — data must be fixed upstream."
            )
        return text

    def _load_stats(self, info: dict) -> Optional[dict]:
        """Load ``meta/<STATS_FILENAME>`` stats when declared, else None.

        Default covers the OXE per-dataset ``eef_stats.json`` (flat dict).
        RoboCOIN overrides for its dynamic per-robot-type stats path.
        """
        if self.STATS_FILENAME is None:
            return None
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir / "meta" / self.STATS_FILENAME
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but {stats_path} is missing. "
                f"Run the matching compute_stats script for {self.DATASET_NAME}, or set normalize_mode=null."
            )
        if stats_path.suffix == ".npy":
            raw = np.load(stats_path, allow_pickle=True).item()
        else:
            import json

            with open(stats_path) as f:
                raw = json.load(f)
        return materialize_eef_stats(
            raw,
            self._normalize_mode,
            dim=self.STATS_DIM,
            strict_minmax=self.STATS_STRICT_MINMAX,
            source_hint=str(stats_path),
        )

    def _write_deploy_normalizer_stats(self, combined: dict, keys) -> None:
        """Write ``meta/normalization_stats.npy`` — the deploy denormalizer artifact.

        ``combined`` is this reader's RAW-space per-mode stats (the values the
        reader normalizes against, BEFORE the unify scatter). The model emits the
        unified action; at deploy the policy server's ``_UnifyAwareNormalizer``
        gathers that unified output back to raw dims, THEN unnormalizes with these
        RAW stats — so the artifact is authored in RAW space (NOT scattered to the
        unified width). Schema is the nested ``{DEPLOY_ACTION_MODE: {mean, std, min,
        max, q01, q99}}`` that ``load_mode_stats`` / ``_build_normalizer`` consume
        unchanged. Shared by every reader that serves the unified action: set
        ``DEPLOY_ACTION_MODE`` and call this from ``_load_stats``.

        The file may legitimately hold MORE than one ``action_mode`` (a bucket used
        by both a ``unified`` and a ``joint`` run): this MERGES this mode's entry into
        whatever the file already holds rather than overwriting. Overwriting would
        drop the other mode's key, and the deploy side (``_build_inner_normalizer``)
        now hard-fails on a missing key — so a blind overwrite would turn a stale
        artifact into a refused deployment. (Merge closes the sequential case; the
        deploy-side raise backstops the residual same-instant two-writer race.)

        Best-effort on the write: a read-only dataset mount raises ``OSError``, which
        is caught and logged — the artifact is skipped (``normalization_stats_path``
        stays ``None``) instead of crashing construction. Training itself never reads
        this file (it normalizes in-process from the reader stats), so a RO mount
        degrades to "no deploy artifact" rather than an ``__init__`` failure.
        """
        if self.DEPLOY_ACTION_MODE is None:
            raise ValueError(
                f"{self.DATASET_NAME}: _write_deploy_normalizer_stats called but DEPLOY_ACTION_MODE is None; "
                "set it on the reader class to the action_mode key the deploy stats are stored under."
            )
        entry = {k: np.asarray(combined[k], dtype=np.float32) for k in keys}
        out = self._dataset_dir / "meta" / "normalization_stats.npy"
        try:
            # Read-modify-write: preserve other modes' keys already in the file.
            # A genuinely corrupt file (truncated / not a pickle) is caught below and
            # rewritten with just this mode. A bare OSError on the READ (e.g. a
            # transient NFS/fuseblk blip) is NOT treated as corruption — it propagates
            # to the outer handler, which skips the write and leaves the existing
            # (valid) file intact rather than clobbering another mode's key with a
            # single-key rewrite (which the deploy-side raise would then reject).
            payload: Dict[str, Any] = {}
            if out.exists():
                try:
                    prev = np.load(out, allow_pickle=True).item()
                    if isinstance(prev, dict):
                        payload.update(prev)
                except (ValueError, EOFError, pickle.UnpicklingError) as e:
                    logger.warning(
                        "%s(%s): existing %s is corrupt (%s); rewriting with only the %r key.",
                        self.DATASET_NAME,
                        self._dataset_id,
                        out.name,
                        e,
                        self.DEPLOY_ACTION_MODE,
                    )
            payload[self.DEPLOY_ACTION_MODE] = entry
            # Atomic write (unique temp + replace) so concurrent per-rank constructors
            # never observe a half-written file. pid ALONE collides on a shared
            # filesystem (same local-rank across nodes → same pid), so the temp name
            # also carries hostname + a uuid. Ends in '.npy' so np.save adds no suffix.
            tmp = out.with_name(f".{out.stem}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.npy")
            try:
                np.save(tmp, payload, allow_pickle=True)
                tmp.replace(out)
            finally:
                if tmp.exists():
                    tmp.unlink()
        except OSError as e:
            # Read-only mount (write) or transient IO fault (read): leave any existing
            # file untouched and skip the artifact. normalization_stats_path stays None
            # so the trainer copy is skipped and deploy fails loud on the missing key.
            logger.warning(
                "%s(%s): could not read/write deploy stats artifact %s (%s); "
                "normalization_stats.npy left unchanged (deploy artifact not (re)generated). "
                "Training is unaffected (in-process normalization uses the reader stats); to "
                "deploy this run, pre-generate the artifact on a writable copy of the meta/ dir.",
                self.DATASET_NAME,
                self._dataset_id,
                out,
                e,
            )
            return
        self.normalization_stats_path = str(out)

    def _action_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        """Return normalized ``(actual_raw_len, ACTION_DIM)`` action, or None (disabled)."""
        return None

    def _proprio_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        """Return normalized ``(1, ACTION_DIM)`` proprio, or None (disabled)."""
        return None

    def _post_init(self, info: dict) -> None:
        """Optional one-time per-dataset setup (e.g. Fractal quat sanity check)."""

    @classmethod
    def _multibucket_wrapper(cls):
        """Root-mode aggregate wrapper class, or None when only single-bucket is supported."""
        return None

    # ----- offsets / IO -----------------------------------------------------

    def _video_cameras(self) -> Tuple[str, ...]:
        """Cameras needing per-episode video frame offsets: the resolved trio
        plus every head-view sampling choice, deduped, order-stable."""
        cams = [self._head_camera, self._left_wrist_camera, self._right_wrist_camera]
        cams.extend(self._head_camera_choices or ())
        return tuple(dict.fromkeys(c for c in cams if c is not None))

    def _add_episode_offsets(self, eps: pd.DataFrame) -> None:
        self._add_data_offsets(eps)
        for cam in self._video_cameras():
            # 这两个是lerobot meta中原始存在的字段
            chunk_col = f"videos/{cam}/chunk_index"
            file_col = f"videos/{cam}/file_index"
            if chunk_col in eps.columns and file_col in eps.columns:
                eps[self._video_offset_col(cam)] = compute_file_local_offsets(eps, chunk_col, file_col)

    @staticmethod
    def _video_offset_col(camera: str) -> str:
        return f"_video_frame_offset/{camera}"

    def _load_data_table(self, chunk_idx: int, file_idx: int):
        path = self._dataset_dir / self._data_path_template.format(chunk_index=chunk_idx, file_index=file_idx)
        return _read_data_table_cached(str(path), tuple(self.NEEDED_COLS))

    # ----- Dataset ----------------------------------------------------------

    def __len__(self) -> int:
        return self._n_total

    def __getitem__(self, idx: int) -> dict:
        return self._safe_get(idx)

    def _safe_get(self, idx: int) -> dict:
        for attempt in range(_GETITEM_MAX_RETRIES):
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                if getattr(self, "_wan_latent_cache", None) is not None:
                    # Never silently substitute a different window on missing/corrupt cache.
                    raise
                if attempt == _GETITEM_MAX_RETRIES - 1:
                    raise
                self._fail_count += 1
                if self._fail_count == 1 or self._fail_count % self._fail_log_every == 0:
                    logger.warning(
                        "%s(%s): %d cumulative __getitem__ failures (latest: idx=%d, %s, attempt=%d)",
                        self.DATASET_NAME,
                        self._dataset_id,
                        self._fail_count,
                        idx,
                        type(e).__name__,
                        attempt,
                    )
                idx = (idx + 1) % max(1, len(self))
        raise RuntimeError("unreachable")

    def _getitem_impl(self, idx: int) -> dict:
        # LINK: openwam/dataloader/bases/lerobot_v3_reader.py:396
        # 根据上面算好的，确定当前 idx 属于哪个 episode
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)      # 注意这里的 -1：_cum_n_starts是累加出来的，要减去 1 才指向当前 episode
        # 这个也是上面算好的，获取这个 episode 的合法起止点
        valid_start = int(self._ep_valid_start[ep_local])
        valid_end = int(self._ep_valid_end[ep_local])
        # 获取 idx 具体在这条 episode 内的位置
        offset = valid_start + (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        row = self._eps_df.iloc[ep_local]

        # NOTE: 这里是处理 padding 逻辑的一部分，这个算出当前窗口起点下，还剩多少真实数据
        # 这个本质上就决定了末尾最多 padding 多少，而再往前推，valid_end 决定最后终止点，而 valid_end
        # LINK: openwam/dataloader/bases/lerobot_v3_reader.py:529
        actual_raw_len = min(self._num_frames, valid_end - offset)

        # 1) parquet window rows
        # LINK: openwam/dataloader/bases/lerobot_v3_reader.py:601
        # episode 在文件内的起点 + 窗口在 episode 内的起点 -> 这个 idx 具体要读 parquet 文件的哪一行
        local_start = int(self._ep_data_row_offset[ep_local]) + offset
        # 读取具体的 data/xxx.parquet, libero 为例，只需要三项 ("action", "observation.state", "task_index")
        # LINK: openwam/dataloader/libero.py:88
        table = self._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))


        # NOTE: 这个 win 就是具体的 lerobot parquet 里的文件了，结合上面的 actual_raw_len，这里只读真实数据
        win = table.slice(local_start, actual_raw_len).to_pandas()

        # 2) prompt (hook)
        prompt = self._resolve_prompt(row, win)

        # 3) action + 4) proprio (hooks → normalized 20-D payload or None)
        # 内部看具体的数据集实现，libero的仍是 10d action
        # LINK: openwam/dataloader/libero.py:282
        # 外部 finalize 会生成 unified 80d action 和对应 mask (也支持不 unify，就返回原始的)
        # NOTE: 这里也完成了 padding, 这里的 padding 是末尾补 0
        # ! 因此 action_mask 既有 temporal-horizon 上的，又有 action-dimension 上的
        action, action_mask = self._finalize_action(self._action_20d(win), actual_raw_len)
        # ! state 存在时只有 dimension 上的 mask，因为只取当前 step 的一步 state
        proprio, proprio_mask = self._finalize_proprio(self._proprio_20d(win))

        cache = getattr(self, "_wan_latent_cache", None)
        if cache is not None:
            return {
                "video": [], "vace_video": None, "clean_condition_images": None,
                "wan_cached_latent": cache.read(idx), "cache_sample_index": idx,
                "action": torch.from_numpy(action), "action_mask": torch.from_numpy(action_mask),
                "video_mask": torch.from_numpy(self._video_sample_indices < actual_raw_len),
                "proprio": torch.from_numpy(proprio).float(),
                "proprio_mask": torch.from_numpy(proprio_mask), "prompt": prompt,
            }

        # 5) video — decode head (+ optional wrist) frames into the canvas
        # 读取具体video，主视角 reshape 到 256*320，腕部 reshape 到 128*160
        # 末尾 padding 时重复最后一帧
        # 多视角会拼图，不足三视角的用黑色补全。单视角不拼，直接返回
        # 对拼好的图整体做 color jitter，但是黑色会保留
        real_local_indices = self._video_sample_indices[self._video_sample_indices < actual_raw_len]
        reference_offsets = getattr(self, "_clean_condition_frame_offsets", None)
        if reference_offsets is None:
            reference_offsets = condition_frame_offsets(self._history_num_tubelets)
        if len(reference_offsets) > 1:
            # Clamp within this episode, before adding the camera's file offset.
            history_indices = [max(offset + delta, 0) - offset
                               for delta in reference_offsets[:-1]]
            real_local_indices = np.concatenate((history_indices, real_local_indices))
        if self._color_jitter is not None:
            # Same jitter factors across the whole clip (temporal consistency).
            video, exclusion_masks = self._decode_window_video(
                row,
                ep_local,
                offset,
                real_local_indices,
                idx,
                return_missing_masks=True,
            )
            jitter_input = {"video": video}
            if exclusion_masks is not None:
                jitter_input["video_jitter_exclusion_masks"] = exclusion_masks
            video = self._color_jitter.apply(jitter_input)["video"]
        else:
            video = self._decode_window_video(row, ep_local, offset, real_local_indices, idx)
        video_mask = torch.from_numpy(self._video_sample_indices < actual_raw_len)
        if len(reference_offsets) > 1:
            num_history_images = len(reference_offsets) - 1
            reference_frames = video[:num_history_images + 1]
            video = video[num_history_images:]
        else:
            reference_frames = [video[0]] if video else []

        return {
            "video": video,
            "vace_video": None,
            "clean_condition_images": reference_frames,
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "video_mask": video_mask,
            "proprio": torch.from_numpy(proprio).float(),
            "proprio_mask": torch.from_numpy(proprio_mask),
            "prompt": prompt,
        }

    # ----- action / proprio finalization -----------------------------------

    def _finalize_action(self, action_20d: Optional[np.ndarray], actual_raw_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad an action payload to ``(T_action, ACTION_DIM)`` + build its 2-D mask.

        ``action_20d is None`` → all-zero action + all-False mask (video-only
        / disabled supervision). When unify is on, the raw ``_raw_action_dim``
        payload is scattered into ``unify_dim`` slots (mask follows the mapped
        dims) — this happens AFTER normalization (``action_20d`` is already
        normalized by ``_action_20d``).
        """
        T_action = self._num_frames - 1
        width = self._raw_action_dim  # fill at raw width first; unify-scatter below
        # ! 末尾补 0 padding
        action = np.zeros((T_action, width), dtype=np.float32)
        n_valid = 0
        n_supervised = 0
        if action_20d is not None:
            n_valid = min(actual_raw_len, T_action)
            if n_valid > 0:
                action[:n_valid] = action_20d[:n_valid]
            # Steps with a REAL supervised target. Default == n_valid (row-aligned
            # readers); a reader that shifts the target +1 frame overrides
            # _n_supervised_action_steps so its clamped final boundary step is masked.
            n_supervised = min(self._n_supervised_action_steps(actual_raw_len), T_action)

        if self._unify:
            # (T, raw) -> (T, unify_dim). The dim mask is the precomputed
            # ACTION_DIM_MASK-honoring one (map_to_unify's own mask would mark
            # every mapped slot valid, leaking single-arm right-arm padding).
            action, _ = map_to_unify(action, self._unify_dst_index, self._unify_dim)
            dim_mask = self._unify_dim_mask
        else:
            dim_mask = self.ACTION_DIM_MASK

        action_mask = build_action_mask_2d(
            T_action=T_action,
            action_dim=self.ACTION_DIM,
            n_valid_time=n_supervised,
            dim_mask=dim_mask,
        )
        return action, action_mask

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        """Number of action steps in the window that carry a REAL supervised target.

        Default: every row present (``actual_raw_len``) — row-aligned readers read
        the action at row ``t`` directly, so all rows are real. A reader that builds
        the target by shifting the achieved pose +1 frame (so the last row of a
        boundary window is a clamped / fabricated target) overrides this to drop
        that final step (e.g. ``actual_raw_len`` if the window is full, else
        ``actual_raw_len - 1``). The result is min-capped to ``T_action`` by the
        caller, so the default reproduces the previous ``n_valid`` exactly.
        """
        return actual_raw_len

    def _finalize_proprio(self, proprio_20d: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Build ``(1, ACTION_DIM)`` proprio + its 2-D mask.

        ``proprio_20d is None`` → all-zero proprio + all-False mask. When unify
        is on, the raw proprio is scattered into ``unify_dim`` slots (same map
        as the action), after normalization.
        """
        if proprio_20d is None:
            proprio = np.zeros((1, self.ACTION_DIM), dtype=np.float32)
            mask = build_proprio_mask_2d(action_dim=self.ACTION_DIM, enabled=False)
            return proprio, mask

        proprio = np.asarray(proprio_20d, dtype=np.float32)
        if self._unify:
            # Same scatter as action, but honor the proprio-specific mask when
            # the source's action/state field availability differs.
            proprio, _ = map_to_unify(proprio, self._unify_state_dst_index, self._unify_dim)
            dim_mask = self._unify_proprio_dim_mask
        else:
            dim_mask = self.PROPRIO_DIM_MASK
            if dim_mask is None:
                dim_mask = self.ACTION_DIM_MASK

        mask = build_proprio_mask_2d(
            action_dim=self.ACTION_DIM,
            enabled=True,
            dim_mask=dim_mask,
        )
        return proprio, mask

    # ----- video ------------------------------------------------------------

    def _decode_window_video(
        self,
        row,
        ep_local: int,
        offset: int,
        real_local_indices: np.ndarray,
        idx: int,
        *,
        return_missing_masks: bool = False,
    ):
        """Decode the window's frames into a list of PIL images (single-view) or
        L-shape multiview canvases, with last-real-frame padding. Optionally
        return per-frame masks for structurally missing multiview slots.

        Head decode failure is fatal (raises → ``_safe_get`` retries). Wrist
        (auxiliary) decode failure is tolerated (black slot)."""
        head_camera = self._head_camera
        if self._head_camera_choices is not None:
            # Head-view sampling (train only): decode the head slot from one
            # uniformly sampled choice. The frames still land under the
            # _head_camera layout key below — slot names are fixed, only the
            # decoded content varies.
            head_camera = self._head_camera_choices[random.randrange(len(self._head_camera_choices))]
        head_frames = self._decode_one_camera(head_camera, row, ep_local, offset, real_local_indices, kind="head")
        if not head_frames:
            raise RuntimeError(f"empty head-video decode for {self.DATASET_NAME}({self._dataset_id}) at idx={idx}")

        if self._multiview:
            left_frames = (
                self._decode_one_camera(
                    self._left_wrist_camera, row, ep_local, offset, real_local_indices, kind="wrist"
                )
                if self._left_wrist_camera
                else []
            )
            right_frames = (
                self._decode_one_camera(
                    self._right_wrist_camera, row, ep_local, offset, real_local_indices, kind="wrist"
                )
                if self._right_wrist_camera
                else []
            )
        else:
            left_frames = []
            right_frames = []

        # History is decoded/augmented with the same views as the current window.
        offsets = getattr(self, "_clean_condition_frame_offsets", None)
        num_history_frames = len(offsets) - 1 if offsets is not None else max(0, 2 * self._history_num_tubelets - 1)
        num_video_frames = self._num_video_frames + num_history_frames
        # Pad the future tail using the last real head frame.
        n_real = len(head_frames)
        if n_real < num_video_frames:
            pad = num_video_frames - n_real
            head_frames = head_frames + [head_frames[-1]] * pad
            if left_frames:
                left_frames = left_frames + [left_frames[-1]] * pad
            if right_frames:
                right_frames = right_frames + [right_frames[-1]] * pad

        if not self._multiview:
            if return_missing_masks:
                return head_frames, None
            return head_frames

        video = []
        missing_masks = [] if return_missing_masks else None
        for fi in range(num_video_frames):
            frames_dict: Dict[str, Any] = {self._head_camera: head_frames[fi]}
            if left_frames and self._left_wrist_camera:
                frames_dict[self._left_wrist_camera] = left_frames[fi]
            if right_frames and self._right_wrist_camera:
                frames_dict[self._right_wrist_camera] = right_frames[fi]
            if self._seperate_encode_view:
                assembled = assemble_two_view_layout(
                    head_frames[fi], left_frames[fi] if left_frames else None,
                    return_missing_mask=return_missing_masks,
                )
            else:
                assembled = assemble_multiview_layout(
                    frames_dict,
                    self._camera_layout,
                    out_h=self._height,
                    out_w=self._width,
                    return_missing_mask=return_missing_masks,
                )
            if return_missing_masks:
                frame, missing_mask = assembled
                video.append(frame)
                missing_masks.append(missing_mask)
            else:
                video.append(assembled)
        if return_missing_masks:
            return video, missing_masks
        return video

    def _decode_one_camera(self, camera, row, ep_local, offset, real_local_indices, *, kind: str) -> List:
        """Decode one camera's frames. ``kind='head'`` is fatal-on-error; ``kind='wrist'``
        tolerates a missing / corrupt clip by returning ``[]`` (black slot)."""
        if camera is None:
            return []
        chunk_col = f"videos/{camera}/chunk_index"
        file_col = f"videos/{camera}/file_index"
        if chunk_col not in row.index:
            return []
        if camera not in self._ep_video_frame_offsets:
            return []

        if kind == "head":
            path = self._dataset_dir / self._video_path_template.format(
                video_key=camera, chunk_index=int(row[chunk_col]), file_index=int(row[file_col])
            )
            v_base = int(self._ep_video_frame_offsets[camera][ep_local]) + offset
            frame_indices = (real_local_indices + v_base).tolist()
            if self._seperate_encode_view:
                return _decode_video_frames(str(path), frame_indices, 224, 224, resample=Image.Resampling.BILINEAR)
            h, w = (_HEAD_SLOT_H, _HEAD_SLOT_W) if self._multiview else (self._height, self._width)
            return _decode_video_frames(str(path), frame_indices, h, w)

        # wrist: tolerated decode — path/offset computation lives INSIDE the try so a
        # corrupt chunk/file index or offset lookup degrades to a black slot rather
        # than failing the whole sample (matches the pre-refactor robocoin behavior).
        try:
            path = self._dataset_dir / self._video_path_template.format(
                video_key=camera, chunk_index=int(row[chunk_col]), file_index=int(row[file_col])
            )
            v_base = int(self._ep_video_frame_offsets[camera][ep_local]) + offset
            frame_indices = (real_local_indices + v_base).tolist()
            if self._seperate_encode_view:
                return _decode_video_frames(str(path), frame_indices, 224, 224, resample=Image.Resampling.BILINEAR)
            return _decode_video_frames(str(path), frame_indices, _WRIST_SLOT_H, _WRIST_SLOT_W)
        except self.WRIST_DECODE_TOLERATED as e:
            self._wrist_fail_count += 1
            if self._wrist_fail_count == 1 or self._wrist_fail_count % self._fail_log_every == 0:
                logger.warning(
                    "%s(%s): %d cumulative wrist decode failures (latest: %s, camera=%s)",
                    self.DATASET_NAME,
                    self._dataset_id,
                    self._wrist_fail_count,
                    type(e).__name__,
                    camera,
                )
            return []

    # ----- BaseDataset ------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        # Samples leave the reader pre-normalized; downstream layers see None.
        return None

    # ----- from_config ------------------------------------------------------

    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = (
        "num_frames",
        "video_stride",
        "history_num_tubelets",
        "clean_condition_frame_offsets",
        "seperate_encode_view",
        "window_stride",
        "height",
        "width",
        "multiview",
        "normalize_mode",
        "target_camera",
        "camera_layout",
        "head_camera_choices",
        "unify_action",
        "unify_action_map",
        "unify_state_map",
        "color_jitter",
    )

    @classmethod
    def from_config(cls, config, split: str = "train") -> "BaseDataset":
        from openwam.dataloader.utils import get_cfg as _get

        dataset_dir = _get(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError(f"{cls.__name__}: missing dataset_dir")
        root = Path(dataset_dir)

        common: Dict[str, Any] = {"split": split, "history_num_tubelets": history_from_config(config)}
        for key in cls.CONFIG_KEYS:
            v = _get(config, key, _CONFIG_MISSING)
            if v is _CONFIG_MISSING:
                continue
            # ``normalize_mode`` uses key-presence semantics: an explicit
            # ``null`` in the config must reach the ctor as None (disable
            # normalization) instead of being dropped — otherwise the ctor's
            # _NORMALIZE_MODE_UNSET sentinel falls back to DEFAULT_NORMALIZE_MODE
            # (e.g. OXE "quantile"). Other keys keep treating None as "unset".
            if v is None and key != "normalize_mode":
                continue
            common[key] = v

        # Single-bucket mode.
        if (root / "meta" / "info.json").is_file():
            kwargs = dict(common)
            dataset_id = _get(config, "dataset_id")
            if dataset_id is not None:
                kwargs["dataset_id"] = dataset_id
            # Honor total_hours in single-bucket mode too (subsample this bucket
            # to the budget) — otherwise a yaml-level total_hours would be
            # silently ignored when dataset_dir points at one bucket.
            total_hours = _get(config, "total_hours")
            if total_hours is not None:
                kwargs["max_hours"] = float(total_hours)
                kwargs["subsample_seed"] = int(_get(config, "seed", 42))
            return cls(dataset_dir=str(root), **kwargs)

        # Root mode (only when the reader declares a multi-bucket wrapper).
        wrapper = cls._multibucket_wrapper()
        if wrapper is None:
            raise FileNotFoundError(f"{cls.__name__}: {root}/meta/info.json missing")
        if not root.is_dir():
            raise FileNotFoundError(f"{cls.__name__}: {root} does not exist")
        sub_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "meta" / "info.json").is_file())
        if not sub_dirs:
            raise FileNotFoundError(f"{cls.__name__}: no sub-buckets with meta/info.json under {root}")
        logger.info("%s.from_config: root mode, %d buckets under %s", cls.__name__, len(sub_dirs), root)
        base_seed = int(_get(config, "seed", 42))
        total_hours = _get(config, "total_hours")
        return build_multibucket(
            cls,
            sub_dirs,
            common,
            base_seed=base_seed,
            total_hours=total_hours,
            wrapper_cls=wrapper,
            source_name=cls.__name__,
        )


__all__ = ["LeRobotV3Reader"]
