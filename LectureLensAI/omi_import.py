
import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

load_dotenv(override=True)

OMI_URL = "https://api.omi.me/v1/dev/user/conversations"
COLLECTION = "lecturelens_memory"


def get_omi_conversations():
    api_key = os.getenv("OMI_API_KEY")
    if not api_key:
        raise ValueError("OMI_API_KEY is missing from .env")

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(
        OMI_URL,
        headers=headers,
        params={"include_transcript": "true"},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if isinstance(data, dict):
        conversations = data.get("conversations", data.get("items", []))
    else:
        conversations = data

    if not isinstance(conversations, list):
        raise ValueError("Unexpected Omi conversations response format")

    return conversations, headers


def get_transcript(conversation, headers):
    segments = conversation.get("transcript_segments") or []

    # Fetch full conversation if the list response has no transcript.
    if not segments and conversation.get("id"):
        response = requests.get(
            f"{OMI_URL}/{conversation['id']}",
            headers=headers,
            params={"include_transcript": "true"},
            timeout=30,
        )
        response.raise_for_status()
        conversation = response.json()
        segments = conversation.get("transcript_segments") or []

    text = "\n".join(
        f"{segment.get('speaker_name') or 'Speaker'}: {segment.get('text', '').strip()}"
        for segment in segments
        if segment.get("text", "").strip()
    )

    # Some responses may expose a full-text field instead.
    if not text.strip():
        text = conversation.get("text", "").strip()

    structured = conversation.get("structured") or {}
    title = structured.get("title") or "Omi Lecture"

    return conversation, title, text


def save_omi_to_qdrant(conversation, title, transcript):
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_key:
        raise ValueError("Qdrant settings are missing from .env")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    embedder = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Split long transcripts into overlapping chunks.
    chunk_size = 1000
    overlap = 150
    chunks = []
    start = 0

    while start < len(transcript):
        chunk = transcript[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(transcript):
            break
        start += chunk_size - overlap

    vectors = list(embedder.embed(chunks))
    points = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "title": title,
                    "session": conversation.get("id", ""),
                    "text": chunk,
                    "chunk_index": index,
                    "source": "omi",
                    "omi_conversation_id": conversation.get("id", ""),
                    "created_at": conversation.get("created_at", ""),
                },
            )
        )

    if points:
        client.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True,
        )

    return len(points)


def render_omi_import():
    st.subheader("Import Omi Conversations")
    st.caption("Import a saved Omi transcript into LectureLens AI memory.")

    if st.button("Fetch Omi Conversations"):
        try:
            conversations, _ = get_omi_conversations()
            st.session_state["omi_conversations"] = conversations
            st.success(f"Found {len(conversations)} conversations.")
        except Exception as exc:
            st.error(f"Could not fetch Omi conversations: {exc}")

    conversations = st.session_state.get("omi_conversations", [])

    if conversations:
        options = {}
        for conversation in conversations:
            cid = conversation.get("id", "")
            structured = conversation.get("structured") or {}
            title = structured.get("title") or "Untitled conversation"
            options[f"{title} — {cid}"] = conversation

        selected_label = st.selectbox(
            "Choose a conversation",
            list(options.keys()),
        )

        if st.button("Import Selected Conversation to Qdrant"):
            try:
                conversation, headers = options[selected_label], {
                    "Authorization": f"Bearer {os.getenv('OMI_API_KEY')}"
                }
                full_conversation, title, transcript = get_transcript(
                    conversation, headers
                )

                if not transcript.strip():
                    st.warning(
                        "No transcript found. Check that Omi has finished processing this recording."
                    )
                else:
                    count = save_omi_to_qdrant(
                        full_conversation, title, transcript
                    )
                    st.success(
                        f"Imported {count} transcript chunks into Qdrant."
                    )
                    st.text_area(
                        "Transcript preview",
                        transcript[:2000],
                        height=180,
                    )
            except Exception as exc:
                st.error(f"Import failed: {exc}")
