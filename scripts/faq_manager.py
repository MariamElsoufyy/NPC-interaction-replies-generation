"""
Interactive FAQ manager — seed, list, view, update, delete.

Usage:
    python -m scripts.faq_manager
"""

import asyncio
import io
import json
import os
import re
import sys
import uuid

import httpx
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.characters import characters_info
from app.characters.build_prompt import build_prompts
from app.core import config
from app.core.clients import AIClients
from app.db.database import get_engine, get_session_factory
from app.db.repositories.faq_repository import (
    create_faq,
    delete_faq,
    get_all_faqs,
    get_faq_by_id,
    update_faq,
)
from app.services.embedding_service import generate_embedding
from app.services.llm.groq_service import LLMGroqService

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
STORAGE_BUCKET = os.getenv("SUPABASE_AUDIO_BUCKET", "faq-audios")


# ── UI helpers ─────────────────────────────────────────────────────────────────

def inp(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {label}{suffix}: ").strip()
    return val or default


def choose(label: str, options: list[str]) -> str:
    print(f"\n  {label}")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        raw = input("  Choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice.")


def confirm(label: str) -> bool:
    return input(f"\n  {label} (y/n): ").strip().lower() == "y"


def header(title: str):
    print("\n" + "━" * 55)
    print(f"  {title}")
    print("━" * 55)


def divider():
    print("\n" + "─" * 55)


# ── Display ────────────────────────────────────────────────────────────────────

def print_faq(faq, index: int = None):
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n  {prefix}ID        : {faq.id}")
    print(f"     Character : {faq.character_id}")
    print(f"     Language  : {faq.language}")
    print(f"     Tag       : {faq.tag or '—'}")
    print(f"     Question  : {faq.question}")
    print(f"     Answer    : {faq.answer[:80]}{'...' if len(faq.answer) > 80 else ''}")
    print(f"     Audio     : {'✅' if faq.audio_url else '❌ none'}")
    print(f"     Embedding : {'✅' if faq.embedding is not None else '❌ none'}")


# ── Audio helpers ──────────────────────────────────────────────────────────────

def generate_audio_bytes(text: str, character_id: str, elevenlabs_client) -> bytes | None:
    import librosa
    voice_id = characters_info.voices.get(character_id.lower())
    if not voice_id:
        print(f"  ⚠️  No voice ID configured for character {character_id}.")
        return None

    print("  ⏳ Generating audio via ElevenLabs...")
    mp3_chunks = []
    stream = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=config.ELEVENLABS_MODEL_ID,
        output_format="mp3_44100_128",
        voice_settings=config.VOICE_SETTINGS,
    )
    for chunk in stream:
        if chunk:
            mp3_chunks.append(chunk)

    if not mp3_chunks:
        print("  ❌ ElevenLabs returned no audio.")
        return None

    mp3_bytes = b"".join(mp3_chunks)
    audio, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
    trimmed, _ = librosa.effects.trim(audio, top_db=35, frame_length=512, hop_length=128)
    if len(trimmed) == 0:
        trimmed = audio

    wav_buf = io.BytesIO()
    sf.write(wav_buf, trimmed, sr, format="WAV", subtype="PCM_16")
    print(f"  ✅ Audio generated ({len(trimmed)} samples @ {sr}Hz)")
    return wav_buf.getvalue()


async def upload_audio(audio_bytes: bytes, character_id: str) -> str | None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("  ⚠️  SUPABASE_URL or SUPABASE_SERVICE_KEY not set.")
        return None

    filename = f"{character_id.lower()}_{uuid.uuid4().hex[:8]}.wav"
    url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{filename}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "audio/wav",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, content=audio_bytes, headers=headers)

    if response.status_code in (200, 201):
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{filename}"
        print(f"  ✅ Uploaded → {public_url}")
        return public_url
    else:
        print(f"  ❌ Upload failed: {response.status_code} — {response.text}")
        return None


# ── LLM helper ─────────────────────────────────────────────────────────────────

def llm_generate_answer(question: str, character_id: str) -> str:
    print("\n  ⏳ Generating answer with LLM...")
    clients = AIClients().get_all_clients()
    llm = LLMGroqService(client=clients["groq_client"])
    prompt_key = config.get_prompt_key_by_character_id(character_id)
    user_prompt, system_prompt = build_prompts(
        character_id=character_id,
        question=question,
        prompt_key=prompt_key,
    )
    raw = llm.generate_reply(user_prompt, system_prompt)
    try:
        parsed = json.loads(raw)
        return parsed.get("answer", raw)
    except Exception:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group()).get("answer", raw)
            except Exception:
                pass
    return raw


# ── Resolve FAQ from list ──────────────────────────────────────────────────────

def _resolve_id(raw: str, faqs) -> uuid.UUID | None:
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(faqs):
            return faqs[idx].id
        print("  ❌ Number out of range.")
        return None
    try:
        return uuid.UUID(raw.strip())
    except ValueError:
        print("  ❌ Invalid input.")
        return None


def _pick_faq(faqs) -> uuid.UUID | None:
    for i, faq in enumerate(faqs, 1):
        print(f"  [{i}] {faq.character_id} | {faq.question[:60]}")
    raw = inp("\nEnter number or FAQ ID")
    return _resolve_id(raw, faqs)


# ── Actions ────────────────────────────────────────────────────────────────────

async def action_add(db):
    header("➕  ADD A FAQ")

    # Question
    question = inp("Question")
    if not question:
        print("  Question cannot be empty.")
        return

    # Character
    available = list(characters_info.first_name.keys())
    display = [f"{c} — {characters_info.first_name[c]} {characters_info.last_name[c]}" for c in available]
    character_id = choose("Character:", display).split(" — ")[0]

    # Language & tag
    language = choose("Language:", ["en", "ar"])
    tag = inp("Tag (optional, e.g. 'academics', 'personal')")

    # Answer
    source = choose("Answer source:", ["Generate with LLM", "Type manually"])
    if source == "Generate with LLM":
        answer = llm_generate_answer(question, character_id)
        print(f"\n  💬 LLM answer:\n  {answer}\n")
        if not confirm("Use this answer?"):
            answer = inp("Enter your answer")
    else:
        answer = inp("Answer")

    if not answer:
        print("  Answer cannot be empty.")
        return

    # Audio
    audio_url = None
    if confirm("Generate and upload audio?"):
        clients = AIClients().get_all_clients()
        audio_bytes = generate_audio_bytes(answer, character_id, clients["elevenlabs_client"])
        if audio_bytes:
            audio_url = await upload_audio(audio_bytes, character_id)

    # Embedding
    print("\n  ⏳ Generating embedding...")
    embedding = generate_embedding(question)
    print(f"  ✅ Embedding generated ({len(embedding)} dims)")

    # Save
    print("\n  ⏳ Saving to database...")
    faq = await create_faq(db, {
        "character_id": character_id,
        "question": question,
        "answer": answer,
        "language": language,
        "tag": tag or None,
        "audio_url": audio_url,
        "embedding": embedding,
    })

    header("✅  FAQ SAVED")
    print_faq(faq)


async def action_list(db):
    header("📋  ALL FAQs")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs in database.")
        return
    for i, faq in enumerate(faqs, 1):
        print_faq(faq, index=i)
        divider()
    print(f"\n  Total: {len(faqs)} FAQ(s)")


async def action_view(db):
    header("🔍  VIEW A FAQ")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs in database.")
        return
    faq_id = _pick_faq(faqs)
    if not faq_id:
        return
    faq = await get_faq_by_id(db, faq_id)
    if faq:
        print_faq(faq)
    else:
        print("  ❌ FAQ not found.")


async def action_update(db):
    header("✏️   UPDATE A FAQ")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs to update.")
        return

    faq_id = _pick_faq(faqs)
    if not faq_id:
        return
    faq = await get_faq_by_id(db, faq_id)
    if not faq:
        print("  ❌ FAQ not found.")
        return

    print_faq(faq)

    field = choose("What do you want to update?", [
        "Question",
        "Answer",
        "Language",
        "Tag",
        "Regenerate embedding",
        "Regenerate audio",
        "Update answer + regenerate embedding + regenerate audio",
        "Cancel",
    ])

    if field == "Cancel":
        return

    updates = {}
    clients_cache = {}

    def get_clients():
        if not clients_cache:
            clients_cache["c"] = AIClients().get_all_clients()
        return clients_cache["c"]

    if field == "Question":
        new_q = inp("New question", default=faq.question)
        updates["question"] = new_q
        if confirm("Regenerate embedding for new question?"):
            print("  ⏳ Generating embedding...")
            updates["embedding"] = generate_embedding(new_q)
            print(f"  ✅ Done ({len(updates['embedding'])} dims)")

    elif field == "Answer":
        source = choose("Answer source:", ["Generate with LLM", "Type manually"])
        if source == "Generate with LLM":
            new_a = llm_generate_answer(faq.question, faq.character_id)
            print(f"\n  💬 LLM answer:\n  {new_a}\n")
            if not confirm("Use this answer?"):
                new_a = inp("Enter your answer")
        else:
            new_a = inp("New answer", default=faq.answer)
        updates["answer"] = new_a
        if confirm("Regenerate audio for new answer?"):
            audio_bytes = generate_audio_bytes(new_a, faq.character_id, get_clients()["elevenlabs_client"])
            if audio_bytes:
                audio_url = await upload_audio(audio_bytes, faq.character_id)
                if audio_url:
                    updates["audio_url"] = audio_url

    elif field == "Language":
        updates["language"] = choose("Language:", ["en", "ar"])

    elif field == "Tag":
        updates["tag"] = inp("New tag", default=faq.tag or "") or None

    elif field == "Regenerate embedding":
        print("  ⏳ Generating embedding...")
        updates["embedding"] = generate_embedding(faq.question)
        print(f"  ✅ Done ({len(updates['embedding'])} dims)")

    elif field == "Regenerate audio":
        audio_bytes = generate_audio_bytes(faq.answer, faq.character_id, get_clients()["elevenlabs_client"])
        if audio_bytes:
            audio_url = await upload_audio(audio_bytes, faq.character_id)
            if audio_url:
                updates["audio_url"] = audio_url

    elif field == "Update answer + regenerate embedding + regenerate audio":
        source = choose("Answer source:", ["Generate with LLM", "Type manually"])
        if source == "Generate with LLM":
            new_a = llm_generate_answer(faq.question, faq.character_id)
            print(f"\n  💬 LLM answer:\n  {new_a}\n")
            if not confirm("Use this answer?"):
                new_a = inp("Enter your answer")
        else:
            new_a = inp("New answer", default=faq.answer)
        updates["answer"] = new_a

        print("  ⏳ Generating embedding...")
        updates["embedding"] = generate_embedding(faq.question)
        print(f"  ✅ Embedding done")

        audio_bytes = generate_audio_bytes(new_a, faq.character_id, get_clients()["elevenlabs_client"])
        if audio_bytes:
            audio_url = await upload_audio(audio_bytes, faq.character_id)
            if audio_url:
                updates["audio_url"] = audio_url

    if not updates:
        print("  Nothing to update.")
        return

    updated = await update_faq(db, faq_id, updates)
    if updated:
        print("\n  ✅ Updated successfully.")
        print_faq(updated)
    else:
        print("  ❌ Update failed.")


async def action_delete(db):
    header("🗑️   DELETE A FAQ")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs to delete.")
        return

    faq_id = _pick_faq(faqs)
    if not faq_id:
        return
    faq = await get_faq_by_id(db, faq_id)
    if not faq:
        print("  ❌ FAQ not found.")
        return

    print_faq(faq)
    if not confirm("Permanently delete this FAQ?"):
        print("  Cancelled.")
        return

    deleted = await delete_faq(db, faq_id)
    print("  ✅ Deleted." if deleted else "  ❌ Failed to delete.")


async def action_fill_missing_audio(db):
    header("🎵  FILL MISSING AUDIO")
    faqs = await get_all_faqs(db)
    missing = [f for f in faqs if not f.audio_url]

    if not missing:
        print("  ✅ All FAQs already have audio.")
        return

    print(f"  Found {len(missing)} FAQ(s) without audio:\n")
    for i, faq in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {faq.character_id} | {faq.question[:60]}")

    if not confirm(f"\nGenerate and upload audio for all {len(missing)} FAQs?"):
        print("  Cancelled.")
        return

    clients = AIClients().get_all_clients()
    success, failed = 0, 0

    for i, faq in enumerate(missing, 1):
        print(f"\n  ── [{i}/{len(missing)}] {faq.question[:60]}")
        audio_bytes = generate_audio_bytes(faq.answer, faq.character_id, clients["elevenlabs_client"])
        if not audio_bytes:
            print(f"  ⚠️  Skipped — no voice configured for {faq.character_id}")
            failed += 1
            continue

        audio_url = await upload_audio(audio_bytes, faq.character_id)
        if audio_url:
            await update_faq(db, faq.id, {"audio_url": audio_url})
            success += 1
        else:
            failed += 1

    divider()
    print(f"\n  Done — ✅ {success} uploaded, ❌ {failed} failed.")


async def action_fill_missing_embeddings(db):
    header("🧠  FILL MISSING EMBEDDINGS")
    faqs = await get_all_faqs(db)
    missing = [f for f in faqs if f.embedding is None]

    if not missing:
        print("  ✅ All FAQs already have embeddings.")
        return

    print(f"  Found {len(missing)} FAQ(s) without embeddings:\n")
    for i, faq in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {faq.character_id} | {faq.question[:60]}")

    if not confirm(f"\nGenerate embeddings for all {len(missing)} FAQs?"):
        print("  Cancelled.")
        return

    success, failed = 0, 0

    for i, faq in enumerate(missing, 1):
        print(f"\n  ── [{i}/{len(missing)}] {faq.question[:60]}")
        try:
            embedding = generate_embedding(faq.question)
            await update_faq(db, faq.id, {"embedding": embedding})
            print(f"  ✅ Embedding saved ({len(embedding)} dims)")
            success += 1
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed += 1

    divider()
    print(f"\n  Done — ✅ {success} generated, ❌ {failed} failed.")


# ── Main loop ──────────────────────────────────────────────────────────────────

async def main():
    engine = get_engine()
    session_factory = get_session_factory(engine)

    header("📚  FAQ MANAGER")

    while True:
        action = choose("What would you like to do?", [
            "Add a FAQ",
            "List all FAQs",
            "View a FAQ",
            "Update a FAQ",
            "Delete a FAQ",
            "Fill missing audio (batch)",
            "Fill missing embeddings (batch)",
            "Exit",
        ])

        async with session_factory() as db:
            if action == "Add a FAQ":
                await action_add(db)
            elif action == "List all FAQs":
                await action_list(db)
            elif action == "View a FAQ":
                await action_view(db)
            elif action == "Update a FAQ":
                await action_update(db)
            elif action == "Delete a FAQ":
                await action_delete(db)
            elif action == "Fill missing audio (batch)":
                await action_fill_missing_audio(db)
            elif action == "Fill missing embeddings (batch)":
                await action_fill_missing_embeddings(db)
            elif action == "Exit":
                print("\n  Bye!\n")
                break

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
