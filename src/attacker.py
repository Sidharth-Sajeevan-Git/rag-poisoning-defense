import json, re
from rag import build_index, retrieve, answer

queries = {q["id"]: q for q in json.load(open("data/queries.json"))}
manifest = json.load(open("data/poison_manifest.json"))
col = build_index(("data/clean", "data/poisoned"), "poisoned")
rows, retrieved_n, success_n = [], 0, 0

for p in manifest:
    q = queries[p["qid"]]
    hits = retrieve(col, q["question"], k=3)
    rank = next((h["rank"] for h in hits if h["source"] == p["file"]), None)
    ans = answer(q["question"], hits)
    ok = bool(re.search(p["marker"], ans))
    retrieved_n += rank is not None
    success_n += ok
    rows.append({"class": p["cls"], "qid": p["qid"], "poison_rank": rank,
                 "attack_success": ok, "answer": ans, "hits": hits})
    print(f"{p['cls']:<24}{p['qid']} rank={rank} success={ok} | {ans[:60]}")

n = len(manifest)
print(f"\nPoison retrieval rate: {retrieved_n}/{n}")
print(f"Attack success rate:   {success_n}/{n}")
json.dump(rows, open("results/poisoned.json","w"), indent=2)

