import fitz  # PyMuPDF
import lancedb
from sentence_transformers import SentenceTransformer

def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


# ---- TEST ----

pdf_path = "sample_curriculum.pdf"
text = extract_text(pdf_path)

chunks = chunk_text(text)

print(f"Total Chunks Created: {len(chunks)}\n")
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")
chunk_embeddings = model.encode(chunks)

print("Embedding shape:", chunk_embeddings.shape)
print("First Chunk:\n")
print(chunks[0])
print("Connecting to LanceDB and storing chunks")
db = lancedb.connect("lancedb")

print("Creating table...")
table = db.create_table(
    "curriculum",
    data=[
        {"text": chunks[i], "vector": chunk_embeddings[i]}
        for i in range(len(chunks))
    ],
    mode="overwrite"
)

print("Chunks stored in LanceDB.")
print("\nPerforming retrieval...")

query = "Generate a structured 4-week course plan"
query_embedding = model.encode([query])[0]

results = table.search(query_embedding).limit(3).to_list()

print("\nTop Retrieved Chunks:\n")

for i, result in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(result["text"])
    print("\n")
print("\nPerforming retrieval...")

query = "Generate a structured 4-week course plan"
query_embedding = model.encode([query])[0]

results = table.search(query_embedding).limit(3).to_list()

print("\nTop Retrieved Chunks:\n")

for i, result in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(result["text"])
    print("\n")
print("\nPerforming retrieval...")

query = "Generate a structured 4-week course plan"
query_embedding = model.encode([query])[0]

results = table.search(query_embedding).limit(3).to_list()

print("\nTop Retrieved Chunks:\n")

for i, result in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(result["text"])
    print("\n")
print("\nPerforming retrieval...")

query = "Generate a structured 4-week course plan"
query_embedding = model.encode([query])[0]

results = table.search(query_embedding).limit(3).to_list()

print("\nTop Retrieved Chunks:\n")

for i, result in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(result["text"])
    print("\n")

retrieved_texts = [result["text"] for result in results]
context = "\n\n".join(retrieved_texts)

print("\n--- Context Being Sent to LLM ---\n")
print(context[:1000])