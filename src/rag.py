import pathlib
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
# Project root directory
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
# Models
EMB = SentenceTransformer("all-MiniLM-L6-v2")
MODEL = "llama3.2:3b"
def chunk(text, size=40, overlap=10):
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return out
def build_index(folders=("data/clean",), name="baseline"):
    # Always use the project-level ChromaDB directory
    db_path = BASE_DIR / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    # Rebuild the requested collection
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(
        name,
        metadata={"hnsw:space": "cosine"}
    )
    ids = []
    docs = []
    metas = []
    for folder in folders:
        # Resolve document folder relative to the project root
        folder_path = BASE_DIR / folder
        for f in sorted(folder_path.glob("*.txt")):
            text = f.read_text(encoding="utf-8")
            for n, chunk_text in enumerate(chunk(text)):
                ids.append(f"{folder}-{f.stem}-{n}")
                docs.append(chunk_text)
                metas.append({
                    "source": f.name,
                    "origin": folder
                })
    if docs:
        col.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=EMB.encode(docs).tolist()
        )
    return col
def retrieve(col, q, k=3):
    r = col.query(
        query_embeddings=EMB.encode([q]).tolist(),
        n_results=min(k, col.count())
    )
    return [
        {
            "rank": i + 1,
            "source": m["source"],
            "origin": m["origin"],
            "distance": round(dist, 4),
            "text": d
        }
        for i, (d, m, dist) in enumerate(
            zip(
                r["documents"][0],
                r["metadatas"][0],
                r["distances"][0]
            )
        )
    ]

def answer(q, hits):
    ctx = "\n\n".join(
        f"[{h['source']}] {h['text']}"
        for h in hits
    )
    prompt = (
        "Answer the question using only the context below. "
        "Be brief.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {q}\n"
        "Answer:"
    )
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )
    return r["message"]["content"]


