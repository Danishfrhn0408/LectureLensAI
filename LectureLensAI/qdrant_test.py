
import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

if not url or not api_key:
    raise ValueError("Check QDRANT_URL and QDRANT_API_KEY in .env")

client = QdrantClient(url=url, api_key=api_key)

collection = "lecturelens_memory"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

print("Connecting to Qdrant...")
print(client.get_collections())

# Load embedding model
print("Loading embedding model...")
embedder = TextEmbedding(model_name=model_name)

# Create collection if needed
if not client.collection_exists(collection):
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    )

# Sample lecture
note = (
    "Supervised learning uses labelled data to train a "
    "machine learning model to make predictions."
)

print("Generating embedding...")
vector = list(embedder.embed([note]))[0].tolist()

print("Saving sample lecture note...")
client.upsert(
    collection_name=collection,
    points=[
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": note,
                "session": "demo-lecture-1",
                "type": "lecture_note",
            },
        )
    ],
)

print("Searching lecture memory...")
question = "How does supervised machine learning work?"
query_vector = list(embedder.embed([question]))[0].tolist()

results = client.query_points(
    collection_name=collection,
    query=query_vector,
    limit=3,
    with_payload=True,
).points

for result in results:
    print("Score:", round(result.score, 3))
    print("Note:", result.payload["text"])

print("\nQdrant memory test completed!")
