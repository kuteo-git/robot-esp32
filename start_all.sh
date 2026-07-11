#!/bin/zsh
# Start the whole robot server (3 services)
BASE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$BASE/logs"
echo "Starting Whisper (STT) :8001 ..."
nohup "$BASE/services/run_whisper.sh" > "$BASE/logs/whisper.log" 2>&1 &
echo "Starting VieNeu-TTS    :8002 ..."
nohup "$BASE/services/run_vieneu.sh"  > "$BASE/logs/vieneu.log"  2>&1 &
echo "Starting xiaozhi-server:8000/8003 ..."
nohup "$BASE/run_server.sh" > "$BASE/logs/xiaozhi_server.log" 2>&1 &
sleep 2
echo "Đã khởi động. Xem log trong thư mục logs/. Đợi ~20s cho model nạp xong."
