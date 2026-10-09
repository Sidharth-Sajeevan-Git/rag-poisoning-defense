import json
from rag import build_index, retrieve, answer

col = build_index(("data/clean",), "baseline")
queries = json.load(open("data/queries.json"))
results, correct = [], 0

for q in queries:
    hits = retrieve(col, q["question"], k=3)
    ans = answer(q["question"], hits)
    ok = q["answer"].lower() in ans.lower()
    correct += ok
    results.append({"id": q["id"], "question": q["question"], "expected": q["answer"],
                    "answer": ans, "correct": ok, "hits": hits})
    print(f"{q['id']} {'OK ' if ok else 'BAD'} top={hits[0]['source']} | {ans[:70]}")

json.dump(results, open("results/baseline.json", "w"), indent=2)
print(f"\nBaseline accuracy: {correct}/{len(queries)}")

