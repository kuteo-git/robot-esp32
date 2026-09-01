#!/bin/zsh
# Start the whole robot server.
#
# Everything except Whisper is a launchd agent (~/Library/LaunchAgents/com.user.robot-*.plist),
# so starting it means bootstrapping the agent back into the user domain -- NOT nohup'ing the
# run_*.sh script, which just races launchd and dies with "address already in use".
BASE="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="gui/$(id -u)"
mkdir -p "$BASE/logs"

for f in ~/Library/LaunchAgents/com.user.robot-*.plist; do
  [[ -e "$f" ]] || continue
  label=$(/usr/libexec/PlistBuddy -c "Print :Label" "$f" 2>/dev/null) || continue
  launchctl bootstrap "$DOMAIN" "$f" 2>/dev/null && echo "started $label"
done

# Whisper has no plist -- still started by hand.
if ! lsof -nP -tiTCP:8001 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Starting Whisper (STT) :8001 ..."
  nohup "$BASE/services/run_whisper.sh" > "$BASE/logs/whisper.log" 2>&1 &
fi

sleep 2
echo "Đã khởi động. Xem log trong thư mục logs/. Đợi ~20s cho model nạp xong."
