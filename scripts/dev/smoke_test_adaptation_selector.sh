#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/evaluations/adaptation_smoke

QUERIES=(
  "我想要一个30轮A与B对话，主题是社交媒体为什么容易让人愤怒，现代风格，像播客对谈"
  "我想要一个30轮A与B对话，主题是AI翻译是否真的能解决人与人之间的误解，现代风格"
  "我想要一个30轮A与B对话，主题是家庭关系里为什么困难对话很难开始，温暖但不要鸡汤"
  "我想要一个30轮A与B对话，主题是创伤、自信和身体形象，适合语音训练"
  "我想要一个20轮A与B对话，主题是睡前放松、日常善意和情绪恢复，语气平静"
  "我想要一个30轮A与B对话，主题是人类野心、技术控制和沟通系统失控，现代风格"
)

i=0
for QUERY in "${QUERIES[@]}"; do
  i=$((i + 1))
  echo ""
  echo "=============================="
  echo "TEST $i"
  echo "$QUERY"
  echo "=============================="

  # 如果你的 run_retrieve.sh 支持 query 作为第一个参数，用这个。
  bash scripts/run_retrieve.sh "$QUERY"

  python -m src.build_adaptation_pack \
    --source_pack outputs/source_packs/latest_source_pack.json

  python - <<'PY'
import json
from pathlib import Path

pack_path = Path("outputs/source_packs/latest_adaptation_pack.json")
pack = json.loads(pack_path.read_text(encoding="utf-8"))

primary = pack.get("primary_source") or {}
pt = pack.get("primary_transcript") or {}

print("mode:", pack.get("mode"))
print("primary:", primary.get("title"))
print("score:", primary.get("primary_score"))
print("expanded:", pt.get("expanded"))
print("range:", pt.get("expanded_chunk_range"))
print("primary_chars:", len(pt.get("text", "")))

print("secondary:")
for s in pack.get("secondary_sources", []):
    print(" -", s.get("title"))
    print("   use:", s.get("suggested_use"))

print("ranking:")
for s in pack.get("source_selection_debug", [])[:6]:
    ps = s.get("primary_score", {})
    print(
        " -",
        s.get("title"),
        "| raw=", ps.get("raw_score"),
        "| score=", ps.get("score"),
        "| modern=", ps.get("modern_hits"),
        "| mech=", ps.get("mechanism_hits"),
        "| dist=", s.get("distance"),
    )
PY

  cp outputs/source_packs/latest_source_pack.json \
    "outputs/evaluations/adaptation_smoke/test_${i}_source_pack.json"

  cp outputs/source_packs/latest_adaptation_pack.json \
    "outputs/evaluations/adaptation_smoke/test_${i}_adaptation_pack.json"
done

echo ""
echo "Smoke test completed."
echo "Results saved to outputs/evaluations/adaptation_smoke/"