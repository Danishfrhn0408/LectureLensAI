
import os
import uuid
import requests
import streamlit as st

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

load_dotenv(override=True)

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LectureLens AI",
    page_icon="🎓",
    layout="wide",
)

COLLECTION_NAME = "lecturelens_memory"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OMI_API_URL = "https://api.omi.me/v1/dev/user/conversations"
LYZR_API_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"

# ==========================================
# SERVICE CONNECTIONS
# ==========================================

@st.cache_resource
def get_embedder():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def get_qdrant():
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")

    if not url or not key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")

    client = QdrantClient(url=url, api_key=key)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        raise ValueError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist."
        )

    return client


def split_text(text, chunk_size=1000, overlap=150):
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

        start += chunk_size - overlap

    return chunks


# ==========================================
# SAVE NOTES TO QDRANT
# ==========================================

def save_to_qdrant(title, session, text, source="manual",
                   conversation_id=""):
    if not text.strip():
        raise ValueError("Lecture notes or transcript cannot be empty.")

    client = get_qdrant()
    embedder = get_embedder()
    chunks = split_text(text)

    vectors = list(embedder.embed(chunks))
    points = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "title": title,
                    "session": session,
                    "text": chunk,
                    "chunk_index": index,
                    "source": source,
                    "omi_conversation_id": conversation_id,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)


# ==========================================
# SEARCH LECTURE MEMORY
# ==========================================

def search_lectures(question, limit=5):
    client = get_qdrant()
    embedder = get_embedder()

    query_vector = list(embedder.query_embed(question))[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=limit,
        with_payload=True,
    )

    return results.points


# ==========================================
# LYZR AI
# ==========================================

def ask_lyzr(question, context):
    api_key = os.getenv("LYZR_API_KEY")
    agent_id = os.getenv("LYZR_AGENT_ID")

    if not api_key or not agent_id:
        raise ValueError("LYZR_API_KEY or LYZR_AGENT_ID missing in .env")

    payload = {
        "user_id": "lecturelens_student",
        "agent_id": agent_id,
        "session_id": "lecturelens_streamlit",
        "message": (
            "You are LectureLens AI, a university lecture assistant.\n"
            "Answer using the lecture context below when relevant. "
            "If the context does not contain the answer, clearly say so. "
            "Do not invent lecture facts.\n\n"
            f"LECTURE CONTEXT:\n{context}\n\n"
            f"STUDENT QUESTION:\n{question}"
        ),
    }

    response = requests.post(
        LYZR_API_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()

    # Handle common Lyzr response formats.
    if isinstance(data, dict):
        for key in ("response", "message", "answer", "output"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value

        return str(data)

    return str(data)


# ==========================================
# OMI API
# ==========================================

def get_omi_conversations():
    api_key = os.getenv("OMI_API_KEY")

    if not api_key:
        raise ValueError("OMI_API_KEY missing in .env")

    response = requests.get(
        OMI_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("conversations", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError("Unexpected Omi conversation list format.")


def get_omi_transcript(conversation):
    api_key = os.getenv("OMI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    segments = conversation.get("transcript_segments") or []
    full_conversation = conversation

    # Retrieve full conversation if transcript is not in the list response.
    if not segments:
        conversation_id = conversation.get("id")

        if not conversation_id:
            return full_conversation, ""

        url = f"{OMI_API_URL}/{conversation_id}"

        response = requests.get(
            url,
            headers=headers,
            params={"include_transcript": "true"},
            timeout=30,
        )
        response.raise_for_status()
        full_conversation = response.json()

        if isinstance(full_conversation, dict):
            segments = full_conversation.get("transcript_segments") or []

    transcript_lines = []

    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue

        speaker = (
            segment.get("speaker_name")
            or segment.get("speaker")
            or "Speaker"
        )
        transcript_lines.append(f"{speaker}: {text}")

    transcript = "\n".join(transcript_lines)

    # Fallback if the API returns a direct text field.
    if not transcript:
        transcript = (
            full_conversation.get("transcript")
            or full_conversation.get("text")
            or ""
        )

        if isinstance(transcript, list):
            transcript = "\n".join(
                str(item.get("text", ""))
                if isinstance(item, dict)
                else str(item)
                for item in transcript
            )

    return full_conversation, transcript.strip()


def get_conversation_title(conversation):
    structured = conversation.get("structured") or {}

    if isinstance(structured, dict):
        title = structured.get("title")
        if title:
            return title

    return conversation.get("title") or "Omi Lecture"


# ==========================================
# SIDEBAR
# ==========================================

st.title("🎓 LectureLens AI")
st.caption("Your AI-powered lecture memory assistant.")

with st.sidebar:
    st.header("System status")

    for label, env_name in [
        ("Qdrant", "QDRANT_API_KEY"),
        ("Lyzr", "LYZR_API_KEY"),
        ("Omi", "OMI_API_KEY"),
    ]:
        if os.getenv(env_name):
            st.success(f"{label} credentials configured")
        else:
            st.warning(f"{label} credentials missing")

    st.divider()
    st.subheader("Architecture")
    st.markdown(
        "- Streamlit dashboard\n"
        "- Omi: conversation transcripts\n"
        "- FastEmbed: text embeddings\n"
        "- Qdrant: lecture memory\n"
        "- Lyzr: AI reasoning"
    )

# ==========================================
# DASHBOARD TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "📝 Save Lecture",
    "💬 Ask Past Lectures",
    "🎙️ Import Omi",
])


# ==========================================
# TAB 1: SAVE LECTURE
# ==========================================

with tab1:
    st.header("Save a lecture to your memory")

    with st.form("save_lecture_form"):
        title = st.text_input(
            "Lecture title",
            placeholder="e.g. Introduction to Machine Learning",
        )

        session = st.text_input(
            "Subject or session",
            placeholder="e.g. Machine Learning Week 3",
        )

        notes = st.text_area(
            "Lecture notes or transcript",
            placeholder="Paste your lecture notes or transcript here...",
            height=250,
        )

        submitted = st.form_submit_button("Save to Qdrant")

    if submitted:
        if not title.strip() or not notes.strip():
            st.warning("Please enter a title and lecture notes.")
        else:
            try:
                with st.spinner("Embedding and saving lecture..."):
                    count = save_to_qdrant(
                        title=title.strip(),
                        session=session.strip(),
                        text=notes,
                        source="manual",
                    )

                st.success(f"Saved {count} lecture chunks to Qdrant!")

            except Exception as exc:
                st.error(f"Could not save lecture: {exc}")


# ==========================================
# TAB 2: ASK PAST LECTURES
# ==========================================

with tab2:
    st.header("Ask your past lectures")
    st.write("Ask a question based on notes stored in Qdrant.")

    question = st.text_area(
        "Your question",
        placeholder="e.g. Explain supervised learning from my lectures.",
        height=100,
        key="lecture_question",
    )

    if st.button("Ask LectureLens AI", key="ask_lecture_button"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching lecture memory..."):
                    results = search_lectures(question, limit=5)

                    context_parts = []

                    for result in results:
                        payload = result.payload or {}
                        text = payload.get("text", "")

                        if text:
                            context_parts.append(
                                f"Title: {payload.get('title', 'Untitled')}\n"
                                f"Source: {payload.get('source', 'unknown')}\n"
                                f"Content: {text}"
                            )

                    context = "\n\n---\n\n".join(context_parts)

                if not context:
                    st.warning(
                        "No relevant lecture notes found. Save or import a lecture first."
                    )
                else:
                    with st.spinner("Lyzr AI is preparing your answer..."):
                        answer = ask_lyzr(question, context)

                    st.subheader("Answer")
                    st.markdown(answer)

                    with st.expander("View retrieved lecture context"):
                        st.text(context)

            except Exception as exc:
                st.error(f"Could not answer question: {exc}")


# ==========================================
# TAB 3: IMPORT OMI
# ==========================================

with tab3:
    st.header("Import Omi conversations")
    st.write(
        "Fetch saved Omi conversations and import their transcripts "
        "into your LectureLens AI memory."
    )

    if not os.getenv("OMI_API_KEY"):
        st.warning("Add OMI_API_KEY to your .env file first.")

    if st.button("Fetch Omi Conversations", key="fetch_omi"):
        try:
            with st.spinner("Fetching conversations from Omi..."):
                conversations = get_omi_conversations()

            st.session_state["omi_conversations"] = conversations

            if conversations:
                st.success(f"Found {len(conversations)} conversations.")
            else:
                st.info(
                    "No conversations returned. Check Omi recordings and account."
                )

        except Exception as exc:
            st.error(f"Could not fetch Omi conversations: {exc}")

    conversations = st.session_state.get("omi_conversations", [])

    if conversations:
        options = {}

        for index, conversation in enumerate(conversations):
            conversation_id = conversation.get("id", f"unknown-{index}")
            title = get_conversation_title(conversation)
            label = f"{title} | {conversation_id}"
            options[label] = conversation

        selected_label = st.selectbox(
            "Choose a conversation to import",
            list(options.keys()),
        )

        selected_conversation = options[selected_label]

        if st.button(
            "Import Selected Conversation to Qdrant",
            key="import_omi",
        ):
            try:
                with st.spinner("Retrieving transcript from Omi..."):
                    full_conversation, transcript = get_omi_transcript(
                        selected_conversation
                    )

                if not transcript:
                    st.warning(
                        "No transcript found. Make sure Omi has finished "
                        "processing the recording."
                    )
                else:
                    title = get_conversation_title(full_conversation)
                    conversation_id = str(
                        full_conversation.get("id")
                        or selected_conversation.get("id")
                        or ""
                    )

                    with st.spinner("Saving transcript to Qdrant..."):
                        count = save_to_qdrant(
                            title=title,
                            session=conversation_id,
                            text=transcript,
                            source="omi",
                            conversation_id=conversation_id,
                        )

                    st.success(
                        f"Imported transcript into Qdrant: {count} chunks."
                    )

                    with st.expander("Preview imported transcript"):
                        st.text_area(
                            "Transcript",
                            transcript[:5000],
                            height=250,
                            key="omi_transcript_preview",
                        )

            except Exception as exc:
                st.error(f"Omi import failed: {exc}")

    st.divider()
    st.caption(
        "Imported transcripts become searchable lecture memories. "
        "Avoid importing the same conversation repeatedly to reduce duplicates."
    )
