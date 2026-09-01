#!/bin/zsh
# Stop the robot server.
#
# Most services are launchd agents (~/Library/LaunchAgents/com.user.robot-*.plist) and
# every one of them sets KeepAlive=true, so killing them by PID/port does NOT stop them --
# launchd restarts them within seconds. They have to be booted out of the user domain.
# Whisper (:8001) has no plist and is the one service still started by hand, so it is the
# only one the port sweep is actually for.
DOMAIN="gui/$(id -u)"

for f in ~/Library/LaunchAgents/com.user.robot-*.plist; do
  [[ -e "$f" ]] || continue
  label=$(/usr/libexec/PlistBuddy -c "Print :Label" "$f" 2>/dev/null) || continue
  if launchctl bootout "$DOMAIN/$label" 2>/dev/null; then
    echo "stopped $label"
  fi
done

# Anything left listening that launchd does not own (whisper :8001).
for p in 8000 8001 8002 8003 8004; do
  pid=$(lsof -nP -tiTCP:$p -sTCP:LISTEN 2>/dev/null)
  [[ -n "$pid" ]] && kill $pid 2>/dev/null && echo "stopped port $p (pid $pid)"
done
echo "Đã dừng tất cả."
