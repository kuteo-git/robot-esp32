#!/bin/zsh
# Stop all services by port
for p in 8000 8001 8002 8003; do
  pid=$(lsof -nP -tiTCP:$p -sTCP:LISTEN 2>/dev/null)
  [[ -n "$pid" ]] && kill $pid 2>/dev/null && echo "stopped port $p (pid $pid)"
done
echo "Đã dừng tất cả."
