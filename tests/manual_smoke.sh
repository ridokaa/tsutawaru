#!/usr/bin/env bash
# Manual end-to-end smoke test — no Discord call required.
#
# Speaks Japanese through the system output (which your Multi-Output Device
# routes into BlackHole) while the app captures and transcribes it. This
# exercises the real chain: Core Audio -> BlackHole -> capture -> resample ->
# VAD -> Whisper -> morphology -> romaji -> translation -> console.
#
# Usage:  ./tests/manual_smoke.sh [seconds] [extra run.py flags...]
# Example: ./tests/manual_smoke.sh 60 --provider none --stats

set -uo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
DURATION="${1:-45}"; shift || true
VOICE="${VOICE:-Kyoko}"

if [ ! -x "$PY" ]; then echo "no venv at $PY" >&2; exit 1; fi

# Sanity: is system output routed somewhere BlackHole can see?
OUT=$(system_profiler SPAudioDataType 2>/dev/null | awk \
  '/^        [A-Za-z]/{d=$0; sub(/:$/,"",d); sub(/^ +/,"",d)} \
   /Default Output Device: Yes/{print d}')
echo "system default output: ${OUT:-unknown}"
case "$(echo "$OUT" | tr '[:upper:]' '[:lower:]')" in
  *multi-output*|*blackhole*) ;;
  *) echo "WARNING: output is '$OUT' — BlackHole will receive nothing." >&2
     echo "         menu bar sound icon -> select your Multi-Output Device" >&2 ;;
esac

PHRASES=(
  "おはようございます。いい天気ですね。"
  "今日は日本語を勉強します。"
  "ちょっと待ってください。"
  "昨日は友達と映画を見に行きました。"
  "この番組はとても面白いと思います。"
)

speak() {
  sleep 8                       # let the model warm up before the first phrase
  for p in "${PHRASES[@]}"; do
    say -v "$VOICE" -r 150 "$p"
    sleep 3                     # a gap well past hangover_ms so the VAD closes
  done
}

speak &
SPEAKER=$!
trap 'kill $SPEAKER 2>/dev/null' EXIT

echo "running for ${DURATION}s (voice: $VOICE)..."
"$PY" run.py --device "BlackHole" "$@" &
APP=$!
sleep "$DURATION"
kill -INT $APP 2>/dev/null
wait $APP 2>/dev/null
echo "--- smoke test finished ---"
