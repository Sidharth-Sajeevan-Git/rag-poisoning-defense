import json, re
from rag import build_index, retrieve, answer

INJ = re.compile(r"note to assistant|ignore (all|previous)|when answering|"
                 r"end your answer|you must (respond|answer)|system prompt", re.I)
nums = lambda t: set(re.findall(r"\d+", t))

CONFIGS = {
 "none":        dict(prov=0, inj=0, conf=0, resp=0),
 "provenance":  dict(prov=1, inj=0, conf=0, resp=0),
 "instruction": dict(prov=0, inj=1, conf=0, resp=0),
 "conflict":    dict(prov=0, inj=0, conf=1, resp=0),
 "response":    dict(prov=0, inj=0, conf=0, resp=1),
 "all":         dict(prov=1, inj=1, conf=1, resp=1),
}

def trusted_nums(hits):
    s = set()
    for h in hits:
        if h["origin"] == "data/clean": s |= nums(h["text"])
    return s

def defend(hits, cfg):
    tn, kept, flagged = trusted_nums(hits), [], False
    for h in hits:
        untrusted = h["origin"] != "data/clean"
        bad = ((cfg["prov"] and untrusted) or
               (cfg["inj"] and INJ.search(h["text"])) or
               (cfg["conf"] and untrusted and (nums(h["text"]) - tn)))
        if bad: flagged = True
        else: kept.append(h)
    return kept[:3], flagged

def process(col, q, cfg):
    hits = retrieve(col, q["question"], k=5)
    kept, flagged = defend(hits, cfg)
    ans = answer(q["question"], kept) if kept else "No trusted context."
    if cfg["resp"] and (nums(ans) - trusted_nums(kept)):
        flagged, ans = True, "[blocked]"
    return ans, flagged

