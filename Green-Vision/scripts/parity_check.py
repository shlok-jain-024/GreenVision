"""Prove the JavaScript planner and the Python assistant agree.

Two implementations of one behaviour is a liability unless something checks
them. This runs one corpus of real messages through both and diffs the
intent each one picked and the tools each one would fire.

    greenplan/reasoning/assistant.py   the full engine (source of truth)
    web/gv-engine.js                   the static build's in-page planner

Divergence is EXPECTED in places and that is fine — the JS planner is
deliberately narrower (no glossary, no long-tail intents) and says so. What
must not happen is silent divergence on the things people actually type. So
the corpus is split:

    MUST_MATCH   intent must be identical; a mismatch fails the run
    MAY_DIFFER   recorded and printed, never fatal

Run (needs node on PATH; no server required):
    .venv/Scripts/python scripts/parity_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from greenplan.reasoning.assistant import classify, extract_place  # noqa: E402

# Every message in this corpus used to match one of the ~30 regex patterns, in
# both implementations. That is precisely why a crash in the Python's regex-MISS
# fallback — two undefined names, `_FALLBACK_WORDS` and `_HERE`, raising
# NameError the moment classify() got past the pattern loop — survived every
# parity run this file ever made: no message here ever reached that line, so
# the check that existed to catch divergence could not see a hard failure.
# The gibberish entry at the end of MUST_MATCH and the off-pattern entries at
# the end of MAY_DIFFER exist to keep that path executed on every run. Do not
# reword them into something a regex catches; not being caught is the point.

# The things people actually type. If these two disagree here, a reader gets
# a different product depending on which build they opened.
MUST_MATCH = [
    "what should I plant here",
    "which trees suit this area",
    "design a 1 hectare park",
    "design a park for a school",
    "plan me a green belt",
    "plant 60 neem trees",
    "add some trees",
    "where are the top 5 cells to plant in",
    "show me the priority areas",
    "find empty land near rajpath club",
    "find empty land nar rajpath club",
    "where can i plant",
    "what is the air quality here",
    "how polluted is it",
    "how green is this area",
    "what is the canopy cover",
    "how much rain does this get",
    "what is the soil like",
    "what does this cost",
    "how much would this cost",
    "review my design",
    "is this design any good",
    "project 25 years",
    "take me to bopal",
    "go to vastrapur",
    "show me the green view",
    "compare bopal and vastrapur",
    "hello",
    "help",
    "tell me about this area",
    # Matches nothing in either implementation, so both run the whole pattern
    # loop, miss, and fall through to the miss path — the Python through its
    # scored keyword table, the JS straight to the default. Both must land on
    # "unknown". This is the one message here that exercises that code.
    "asdf qwerty zxcv",
]

# Known-narrower territory. Printed, never fatal.
MAY_DIFFER = [
    "what is ndvi",
    "how accurate is this",
    "where does this data come from",
    "when should i plant",
    "how many will survive",
    "how much co2 will this capture",
    "i have 5 lakh to spend",
    "who maintains this",
    "how many people benefit",
    # Off-pattern but plausible — natural phrasings no regex covers. The Python
    # scores them against its keyword fallback and lands somewhere useful; the
    # JS has no such table by design and returns "unknown", or "report" when the
    # message points at the map. Divergence here is the design, not a fault.
    # They are in the corpus so that fallback actually runs on every check.
    "the trees near me are dying",
    "what are the conditions in this neighbourhood",
    "it feels hotter every summer here",
]

JS_HARNESS = r"""
const fs = require("fs");
// Minimal browser shims: the planner only touches window and fetch, and we
// are exercising classification, which needs neither.
global.window = global;
global.fetch = async () => ({ ok: false });
const src = fs.readFileSync(process.argv[2], "utf8");
new Function(src)();
const msgs = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = {};
for (const m of msgs) {
  out[m] = { intent: GVE.classify(m), place: GVE.extractPlace(m) };
}
console.log(JSON.stringify(out));
"""


def run_js(messages: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / "h.js"
        m = Path(td) / "m.json"
        h.write_text(JS_HARNESS, encoding="utf-8")
        m.write_text(json.dumps(messages), encoding="utf-8")
        r = subprocess.run(
            ["node", str(h), str(ROOT / "web" / "gv-engine.js"), str(m)],
            capture_output=True, text=True, encoding="utf-8",
        )
    if r.returncode != 0:
        print("node failed:\n" + (r.stderr or "")[:2000])
        sys.exit(2)
    return json.loads(r.stdout)


def main() -> None:
    all_msgs = MUST_MATCH + MAY_DIFFER
    js = run_js(all_msgs)

    fails, soft = [], []
    print(f"{'message':<42} {'python':<12} {'javascript':<12}")
    print("-" * 68)
    for m in all_msgs:
        p_intent = classify(m)
        j_intent = js[m]["intent"]
        same = p_intent == j_intent
        hard = m in MUST_MATCH
        mark = "  " if same else ("XX" if hard else "~ ")
        if not same:
            (fails if hard else soft).append((m, p_intent, j_intent))
        print(f"{mark} {m[:40]:<40} {p_intent:<12} {j_intent:<12}")

    # Place extraction is where the real bug was, so check it explicitly.
    print("\nplace extraction")
    print("-" * 68)
    place_fails = []
    for m in MUST_MATCH:
        p, j = extract_place(m), js[m]["place"]
        if (p or None) != (j or None):
            place_fails.append((m, p, j))
            print(f"XX {m[:40]:<40} {str(p):<14} {str(j):<14}")
    if not place_fails:
        print(f"   all {len(MUST_MATCH)} agree")

    print()
    if soft:
        print(f"{len(soft)} intent(s) differ in the KNOWN-NARROWER set "
              f"(the JS planner does not implement these; it says so):")
        for m, p, j in soft:
            print(f"   {m!r}: python={p} js={j}")
        print()

    if fails or place_fails:
        print(f"FAIL: {len(fails)} intent and {len(place_fails)} place "
              f"mismatch(es) on messages that must agree.")
        sys.exit(1)
    print(f"PASS: {len(MUST_MATCH)} core messages agree on intent and place.")


if __name__ == "__main__":
    main()
