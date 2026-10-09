import json, re
from defend_lib import *

queries = json.load(open("data/queries.json"))
manifest = {p["qid"]: p for p in json.load(open("data/poison_manifest.json"))}
pois = build_index(("data/clean", "data/poisoned"), "poisoned")
clean = build_index(("data/clean",), "baseline")
report = {}

for name, cfg in CONFIGS.items():
    succ = det = acc = fp = 0
    for q in queries:
        ans, flagged = process(pois, q, cfg)
        acc += q["answer"].lower() in ans.lower()
        p = manifest.get(q["id"])
        if p:
            det += flagged
            succ += bool(re.search(p["marker"], ans))
    for q in queries:
        _, flagged = process(clean, q, cfg)
        fp += flagged
    report[name] = dict(residual_attacks=succ, detected=det, accuracy=acc, false_positives=fp)
    print(f"{name:<12} attack_success={succ}/6  detected={det}/6  accuracy={acc}/8  false_pos={fp}/8")

json.dump(report, open("results/defense.json", "w"), indent=2)

