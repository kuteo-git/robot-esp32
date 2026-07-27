#!/usr/bin/env bash
# Builds the pool of background beds the robot loops while it is thinking.
#
# The audio itself is NOT in git (config/assets/**/*.wav is ignored, and a repo is the wrong place
# to redistribute music) -- this script is what is committed, so the pool can be rebuilt on any
# machine, or repointed at a different playlist, without hunting for the original files.
#
#   ./services/fetch_thinking_sounds.sh [PLAYLIST_URL] [COUNT]
#
# Each track is cut to one CLIP_SEC segment taken from CLIP_START_S in (intros are usually a slow
# fade from silence, which reads as the sound having failed), levelled so no clip is noticeably
# louder than another, and given short fades so the loop seam does not click. How quiet the bed
# actually plays is NOT baked in here -- that is thinking_loop_gain_db in .config.yaml, so it can
# be tuned without re-downloading.
set -euo pipefail

PLAYLIST="${1:-https://youtube.com/playlist?list=PLIILL6veL783kKkdiIybbxARNY9bAVQYe}"
COUNT="${2:-40}"
OUT_DIR="${THINKING_POOL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/xiaozhi-esp32-server/main/xiaozhi-server/config/assets/thinking/pool}"

CLIP_START_S="${CLIP_START_S:-20}"   # skip the intro fade
CLIP_SEC="${CLIP_SEC:-30}"           # loop length; the wait it covers is ~40-60s
FADE_S="${FADE_S:-0.8}"
RATE="${RATE:-24000}"                # base.py resamples to the connection rate anyway
LOUDNESS="${LOUDNESS:--16}"          # mức đích của clip. TP=-1.5 bên dưới là limiter chống vỡ tiếng

command -v ffmpeg >/dev/null || { echo "cần ffmpeg"; exit 1; }

# Pick the NEWEST yt-dlp on the box, not whatever is first on PATH. YouTube breaks extraction
# often enough that a stale copy fails with "Requested format is not available" while a newer one
# sitting in another env works fine -- which is exactly what happened here (2026.02.04 on PATH,
# 2026.07.04 in the conda base env).
YTDLP=""
best=""
for cand in $(command -v yt-dlp || true) /opt/homebrew/bin/yt-dlp \
            /opt/homebrew/anaconda3/bin/yt-dlp /opt/homebrew/anaconda3/envs/*/bin/yt-dlp; do
  [ -x "$cand" ] || continue
  v="$("$cand" --version 2>/dev/null | head -1)"
  [ -n "$v" ] || continue
  if [ -z "$best" ] || [ "$v" \> "$best" ]; then best="$v"; YTDLP="$cand"; fi
done
[ -n "$YTDLP" ] || { echo "cần yt-dlp"; exit 1; }
echo "yt-dlp   : $YTDLP ($best)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT_DIR"
# Clear the pool first: the server cycles over whatever is in this directory, so leftovers from a
# previous (longer) playlist would stay in rotation forever.
rm -f "$OUT_DIR"/bed*.wav

echo "Playlist : $PLAYLIST"
echo "Lấy      : $COUNT bài -> $OUT_DIR"
echo

# bestaudio/best, not bestaudio: when YouTube serves this session SABR-only there is no audio-only
# stream left and the muxed 360p is the only thing available -- its audio is fine once extracted.
"$YTDLP" --no-update -q --no-warnings --ignore-errors \
       --playlist-items "1-$COUNT" \
       -f "bestaudio/best" -x --audio-format wav \
       -o "$TMP/%(playlist_index)02d.%(ext)s" \
       "$PLAYLIST"

n=0
for f in "$TMP"/*.wav; do
  [ -e "$f" ] || continue
  n=$((n + 1))
  out="$OUT_DIR/bed$(printf '%02d' "$n").wav"
  fade_out_at=$(echo "$CLIP_SEC - $FADE_S" | bc)
  ffmpeg -nostdin -loglevel error -y \
    -ss "$CLIP_START_S" -t "$CLIP_SEC" -i "$f" \
    -af "loudnorm=I=${LOUDNESS}:TP=-1.5:LRA=11,afade=t=in:st=0:d=${FADE_S},afade=t=out:st=${fade_out_at}:d=${FADE_S}" \
    -ac 1 -ar "$RATE" -sample_fmt s16 \
    "$out"
  printf '  %s  %s\n' "$(basename "$out")" "$(du -h "$out" | cut -f1)"
done

echo
echo "Xong: $n clip trong $OUT_DIR"
echo "Trỏ thinking_loop_sound_file tới thư mục này để bật chế độ xoay vòng (không lặp)."
