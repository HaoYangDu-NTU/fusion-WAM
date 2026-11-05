#!/usr/bin/env bash
set -Eeuo pipefail

SRC=${SRC:-/data-team/rhodes_team/huggingface_cache/rhodes_lerobot/TeleData/Tianji/saucer0821_no_idle}
CACHE=${CACHE:-/data/zhangyang/bench_datasets/saucer0821_no_idle_shortgop_g15_v2}
JOBS=${JOBS:-48}

if [[ ! -d "$SRC/videos" || ! -f "$SRC/meta/info.json" ]]; then
  echo "invalid source dataset: $SRC" >&2
  exit 66
fi
if [[ -e "$CACHE" || -L "$CACHE" ]]; then
  echo "cache target already exists; refusing to overwrite: $CACHE" >&2
  exit 73
fi

mkdir -p "$CACHE/videos"
ln -s "$SRC/meta" "$CACHE/meta"
ln -s "$SRC/data" "$CACHE/data"
ln -s "$SRC/meta/info.json" "$CACHE/info.json" 2>/dev/null || true

export SRC CACHE
find "$SRC/videos" -type f -name '*.mp4' -print0 \
  | xargs -0 -r -P "$JOBS" -n 1 bash -c '
      set -Eeuo pipefail
      src=$1
      rel=${src#"$SRC/"}
      dst="$CACHE/$rel"
      mkdir -p "$(dirname "$dst")"
      if [[ -s "$dst" ]]; then
        exit 0
      fi
      # Keep the .mp4 suffix so ffmpeg selects the MP4 muxer for the atomic temp file.
      tmp="$dst.tmp.$BASHPID.mp4"
      trap '\''rm -f "$tmp"'\'' EXIT
      ffmpeg -hide_banner -loglevel error -nostdin -y -i "$src" \
        -map 0:v:0 -an -c:v libx264 -preset ultrafast -crf 18 \
        -g 15 -keyint_min 15 -sc_threshold 0 -pix_fmt yuv420p "$tmp"
      mv -f "$tmp" "$dst"
    ' _

expected=$(find "$SRC/videos" -type f -name '*.mp4' | wc -l)
actual=$(find "$CACHE/videos" -type f -name '*.mp4' | wc -l)
if [[ "$actual" != "$expected" ]]; then
  echo "cache file count mismatch: expected=$expected actual=$actual" >&2
  exit 1
fi
echo "built short-GOP cache: $CACHE ($actual videos)"
