#!/usr/bin/env bash
# Regen thinking-filler wav từ filler_texts.txt qua VieNeu (:8002) -> resample 16k mono.
# Sửa filler_texts.txt rồi chạy lại script này. (Text gốc KHÔNG nhúng trong wav -> txt là nguồn.)
#   ./regen_fillers.sh            # gen tất cả
#   ./regen_fillers.sh 5 12 37    # chỉ gen lại th_5/th_12/th_37 (vd câu VieNeu đọc loạn)
DIR="$(cd "$(dirname "$0")" && pwd)"
VIENEU="${VIENEU_URL:-http://localhost:8002}"
ONLY=" $* "
TMP=/tmp/filler_gen; mkdir -p "$TMP"
n=0
while IFS='|' read -r idx text; do
  [[ "$idx" =~ ^[0-9]+$ ]] || continue
  [ -z "$text" ] && continue
  [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $idx "* ]] && continue
  curl -s -m90 -X POST "$VIENEU/tts" -H "Content-Type: application/json" \
    --data "$(python3 -c "import json,os,sys; print(json.dumps(dict(input=sys.argv[1], voice=os.environ.get('FILLER_VOICE',''))))" "$text")" \
    -o "$TMP/raw_$idx.wav"
  sz=$(stat -f%z "$TMP/raw_$idx.wav" 2>/dev/null || stat -c%s "$TMP/raw_$idx.wav" 2>/dev/null)
  if [ "${sz:-0}" -lt 1000 ]; then echo "th_$idx: LỖI (size $sz) — giữ file cũ"; continue; fi
  ffmpeg -y -i "$TMP/raw_$idx.wav" -ar 16000 -ac 1 "$DIR/th_$idx.wav" >/dev/null 2>&1
  d=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$DIR/th_$idx.wav" 2>/dev/null)
  echo "th_$idx [${d%.*}s]: $text"; n=$((n+1))
done < "$DIR/filler_texts.txt"
echo "=== gen $n file ==="
