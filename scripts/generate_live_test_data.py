"""Generate live-translation test data from Hunter Bible Church sermons.

Pipeline, per sermon (mirrors the sunflower-translation-qa reference project):
  1. Download bestaudio from SoundCloud (yt-dlp) and convert to 16 kHz mono WAV (ffmpeg).
  2. Transcribe by streaming to Deepgram Nova-3 over WebSocket, keeping only
     `final_results` (each: transcript, words[], start, end). Unlike the reference,
     audio is streamed *faster than real-time* (no per-chunk real-time sleep).
  3. Segment the flattened words via the custom intonation-boundary API into
     `{text, start, end}` segments.

Outputs land under live_test_data/:
  audio/<id>.wav
  transcriptions/<id>_transcription.json   (+ <id>_transcription_finals.txt)
  segments/<id>_segments.json              (+ <id>_segments.txt)
  manifest.json                            (the 10 sampled sermons + status)

The Deepgram key is read from the reference project's .env so the secret stays
out of this repo. The segmentation endpoint is public (no key).

Run with the reference project's venv (has yt_dlp / deepgram / websockets / requests / dotenv):
  "/Users/sivakalyan/Programming/Python/Sunflower AI/sunflower-translation-qa/.venv/bin/python" \
      scripts/generate_live_test_data.py
"""

import asyncio
import json
import random
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import websockets
import websockets.exceptions
import yt_dlp
from dotenv import load_dotenv
import os

# --- Config -----------------------------------------------------------------

REF_PROJECT = Path(
    "/Users/sivakalyan/Programming/Python/Sunflower AI/sunflower-translation-qa"
)
load_dotenv(REF_PROJECT / ".env")  # pull DEEPGRAM_API_KEY from the reference project
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

SEGMENTATION_API_URL = (
    "https://sunflower-ai-segmentation-api.azurewebsites.net/api/intonation-boundaries"
)

SOUNDCLOUD_TRACKS_URL = "https://soundcloud.com/hunterbiblechurch/tracks"
NUM_SERMONS = 10
RANDOM_SEED = 20260619  # recorded for reproducibility of the sample

DOWNLOAD_WORKERS = 4  # concurrent yt-dlp downloads (downloads are the bottleneck)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "live_test_data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
SEGMENTS_DIR = DATA_DIR / "segments"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Stream moderately faster than real-time. Deepgram's streaming endpoint is built
# for ~real-time ingestion; blasting audio (e.g. 25x) makes the server drop the
# connection early and truncate the transcript, so we cap the speed-up at ~4x.
CHUNK_SECONDS = 0.25
SEND_SLEEP = 0.0625  # 0.25 s audio per 0.0625 s wall -> ~4x real-time
# If no message arrives for this long, force-close so the receive loop can't block
# forever (Deepgram occasionally fails to close the socket after CloseStream).
IDLE_TIMEOUT = 25.0
OPEN_TIMEOUT = 30.0
# Reject a transcript whose last word ends before this fraction of the audio; it
# means the stream was truncated and the file must be re-transcribed.
MIN_COVERAGE = 0.93
TRANSCRIBE_WORKERS = 4  # concurrent Deepgram streams (independent connections)


# --- Step 1: sampling + download --------------------------------------------


def enumerate_tracks():
    """Return [{id, title, url}, ...] for every track on the account."""
    opts = {"quiet": True, "extract_flat": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(SOUNDCLOUD_TRACKS_URL, download=False)
    return [
        {"id": str(e["id"]), "title": e.get("title", ""), "url": e["url"]}
        for e in info["entries"]
        if e.get("id") and e.get("url")
    ]


def download_audio(track):
    """Download + convert one track to 16 kHz mono WAV. Returns the WAV path."""
    wav_path = AUDIO_DIR / f"{track['id']}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 0:
        print(f"  audio exists, skipping download: {wav_path.name}")
        return wav_path

    tmpl = str(AUDIO_DIR / f"{track['id']}.source.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": tmpl,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([track["url"]])

    sources = list(AUDIO_DIR.glob(f"{track['id']}.source.*"))
    if not sources:
        raise FileNotFoundError(f"No downloaded source for {track['id']}")
    source = sources[0]

    # Convert to the 16 kHz mono 16-bit PCM WAV Deepgram streams happily.
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(wav_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    source.unlink(missing_ok=True)
    return wav_path


def prefetch_audio(sermons, workers=DOWNLOAD_WORKERS):
    """Download all not-yet-present sermon WAVs concurrently.

    Downloads dominate wall-time while transcription is cheap, so we fetch the
    audio in parallel up front, then transcribe/segment serially afterwards.
    Each download writes to per-id paths, so concurrent yt-dlp calls don't clash.
    """
    todo = [s for s in sermons if not (AUDIO_DIR / f"{s['id']}.wav").exists()]
    if not todo:
        print("All audio already downloaded.")
        return
    print(f"Prefetching {len(todo)} audio files with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_audio, s): s for s in todo}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                fut.result()
                print(f"  downloaded: {s['title'][:55]}")
            except Exception as e:
                # Leave it for the serial loop to retry inline; don't abort the batch.
                print(f"  download FAILED {s['id']} ({s['title'][:40]}): {e}")


# --- Step 2: Deepgram streaming transcription -------------------------------


async def _transcribe_async(audio_path, api_key, output_path):
    with wave.open(str(audio_path), "rb") as wav_file:
        channels, sample_width, sample_rate, num_samples, _, _ = wav_file.getparams()
        assert sample_width == 2, "WAV data must be 16-bit"
        audio_data = wav_file.readframes(num_samples)

    byte_rate = sample_width * sample_rate * channels
    chunk_size = int(byte_rate * CHUNK_SECONDS)
    total_duration = num_samples / sample_rate
    print(f"  duration ~{total_duration:.0f}s, {len(audio_data)/1e6:.1f} MB, "
          f"{sample_rate} Hz x{channels}")

    deepgram_url = (
        "wss://api.deepgram.com/v1/listen?"
        "model=nova-3&punctuate=true&utterances=true&"
        f"encoding=linear16&sample_rate={sample_rate}&channels={channels}"
    )

    all_results = []
    metadata = {}
    sent_bytes = 0  # audio successfully streamed so far; the resume point
    max_retries = 12
    failed_attempts = 0

    while sent_bytes < len(audio_data) and failed_attempts < max_retries:
        remaining = audio_data[sent_bytes:]
        seg_offset = sent_bytes / byte_rate  # absolute time at this connection's start
        if failed_attempts > 0:
            print(f"  reconnecting (attempt {failed_attempts + 1}) from {seg_offset:.0f}s...")
            await asyncio.sleep(2)

        # Mutable boxes so the inner coroutines can report back without nonlocal.
        progress = {"sent": 0}
        msgs = {"n": 0}

        try:
            async with websockets.connect(
                deepgram_url,
                additional_headers={"Authorization": f"Token {api_key}"},
                max_size=None,
                open_timeout=OPEN_TIMEOUT,
            ) as ws:
                done_evt = asyncio.Event()

                async def sender(ws):
                    data = remaining
                    try:
                        while data:
                            chunk, data = data[:chunk_size], data[chunk_size:]
                            await ws.send(chunk)
                            progress["sent"] += len(chunk)
                            await asyncio.sleep(SEND_SLEEP)
                        await ws.send(json.dumps({"type": "CloseStream"}))
                    except websockets.exceptions.ConnectionClosed:
                        pass  # server closed early; outer loop resumes from sent_bytes

                async def receiver(ws):
                    # Drain finals until the server actually closes the socket
                    # (which it does after flushing results post-CloseStream).
                    try:
                        async for msg in ws:
                            msgs["n"] += 1
                            res = json.loads(msg)
                            if res.get("request_id"):
                                metadata["request_id"] = res["request_id"]
                            if res.get("is_final"):
                                alt = res.get("channel", {}).get("alternatives", [{}])[0]
                                transcript = alt.get("transcript", "")
                                if transcript:
                                    words = []
                                    for w in alt.get("words", []):
                                        aw = w.copy()
                                        aw["start"] = w.get("start", 0) + seg_offset
                                        aw["end"] = w.get("end", 0) + seg_offset
                                        words.append(aw)
                                    st = words[0]["start"] if words else seg_offset
                                    en = words[-1]["end"] if words else seg_offset
                                    all_results.append({
                                        "transcript": transcript,
                                        "words": words,
                                        "start": st,
                                        "end": en,
                                    })
                                    if output_path:
                                        _save(all_results, metadata, output_path)
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    finally:
                        done_evt.set()

                async def watchdog(ws):
                    """Force-close a silent socket so the receive loop can't hang."""
                    last, idle = -1, 0.0
                    while not done_evt.is_set():
                        await asyncio.sleep(2)
                        if msgs["n"] != last:
                            last, idle = msgs["n"], 0.0
                        else:
                            idle += 2
                            if idle >= IDLE_TIMEOUT:
                                try:
                                    await ws.close()
                                except Exception:
                                    pass
                                return

                await asyncio.gather(
                    sender(ws), receiver(ws), watchdog(ws), return_exceptions=True
                )
        except Exception as e:
            print(f"  connection error: {e}")

        # Advance the resume point by what we actually streamed this connection.
        if progress["sent"] > 0:
            sent_bytes += progress["sent"]
            failed_attempts = 0
        else:
            failed_attempts += 1

    result = {
        "metadata": {"request_id": metadata.get("request_id", "")},
        "final_results": all_results,
    }
    end = all_results[-1]["end"] if all_results else 0.0
    coverage = (end / total_duration * 100) if total_duration else 0
    words = sum(len(r["words"]) for r in all_results)
    print(f"  transcribed: {len(all_results)} finals, {words} words, "
          f"{coverage:.0f}% coverage (ends {end:.0f}s / {total_duration:.0f}s)")
    return result


def _save(results, metadata, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"request_id": metadata.get("request_id", "")},
             "final_results": results},
            f, indent=2, ensure_ascii=False,
        )


def _coverage_ok(audio_path, data, min_cov=MIN_COVERAGE):
    """True if the transcript's last word lands within min_cov of the audio end."""
    fr = data.get("final_results") or []
    if not fr:
        return False
    with wave.open(str(audio_path), "rb") as w:
        dur = w.getnframes() / w.getframerate()
    return fr[-1].get("end", 0) >= min_cov * dur


def transcribe(audio_path, track_id):
    out = TRANSCRIPTIONS_DIR / f"{track_id}_transcription.json"
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        if _coverage_ok(audio_path, data):
            print(f"  transcription exists, skipping: {out.name}")
            return data
        print(f"  re-transcribing (previous was truncated): {out.name}")
    data = asyncio.run(_transcribe_async(audio_path, DEEPGRAM_API_KEY, out))
    _save(data["final_results"], data["metadata"], out)
    finals = TRANSCRIPTIONS_DIR / f"{track_id}_transcription_finals.txt"
    with open(finals, "w", encoding="utf-8") as f:
        for r in data["final_results"]:
            f.write(r.get("transcript", "") + "\n")
    return data


# --- Step 3: segmentation ---------------------------------------------------


def segment(transcription, track_id):
    out_json = SEGMENTS_DIR / f"{track_id}_segments.json"
    tj = TRANSCRIPTIONS_DIR / f"{track_id}_transcription.json"
    # Reuse only if segments are at least as new as the transcription they derive
    # from (so a re-transcribed file gets re-segmented).
    if out_json.exists() and (
        not tj.exists() or out_json.stat().st_mtime >= tj.stat().st_mtime
    ):
        print(f"  segments exist, skipping: {out_json.name}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    words = []
    for fr in transcription.get("final_results", []):
        words.extend(fr.get("words", []))
    if not words:
        raise ValueError("No words to segment")

    api_words, mapping = [], []
    for i, w in enumerate(words):
        pw = w.get("punctuated_word") or w.get("word", "")
        if pw:
            api_words.append(pw)
            mapping.append(i)

    resp = requests.post(SEGMENTATION_API_URL, json={"words": api_words}, timeout=60)
    resp.raise_for_status()
    boundaries = resp.json().get("boundaries", [])
    if len(boundaries) != len(api_words):
        raise ValueError(
            f"boundaries {len(boundaries)} != words {len(api_words)}"
        )

    segments, current = [], []
    for api_idx, (orig_idx, is_boundary) in enumerate(zip(mapping, boundaries)):
        current.append(words[orig_idx])
        if is_boundary or api_idx == len(boundaries) - 1:
            text = " ".join(w.get("punctuated_word", w.get("word", "")) for w in current).strip()
            if text:
                segments.append({
                    "text": text,
                    "start": current[0].get("start", 0.0),
                    "end": current[-1].get("end", 0.0),
                })
            current = []

    out_json.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    (SEGMENTS_DIR / f"{track_id}_segments.txt").write_text(
        "\n".join(s["text"] for s in segments) + "\n", encoding="utf-8"
    )
    print(f"  segmented: {len(segments)} segments")
    return segments


# --- Orchestration ----------------------------------------------------------


def load_or_pick_sample():
    """Pick (and persist) the 10 sermons, or reuse the existing manifest."""
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        print(f"Reusing existing sample of {len(manifest['sermons'])} sermons.")
        return manifest

    print("Enumerating tracks...")
    tracks = enumerate_tracks()
    print(f"  {len(tracks)} tracks found.")
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(tracks, NUM_SERMONS)
    manifest = {
        "source": SOUNDCLOUD_TRACKS_URL,
        "total_tracks": len(tracks),
        "seed": RANDOM_SEED,
        "sermons": [{**t, "status": "pending"} for t in sample],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    if not DEEPGRAM_API_KEY:
        sys.exit("DEEPGRAM_API_KEY not found (checked reference project's .env).")
    for d in (AUDIO_DIR, TRANSCRIPTIONS_DIR, SEGMENTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    manifest = load_or_pick_sample()
    sermons = manifest["sermons"]

    # Phase 1: fetch all audio concurrently (the slow part).
    prefetch_audio(sermons)

    # Phase 2: transcribe in parallel. transcribe() skips files already covered
    # and re-does truncated ones, so this both fills gaps and self-heals.
    need = [s for s in sermons if not _coverage_ok(
        AUDIO_DIR / f"{s['id']}.wav",
        json.loads((TRANSCRIPTIONS_DIR / f"{s['id']}_transcription.json").read_text())
        if (TRANSCRIPTIONS_DIR / f"{s['id']}_transcription.json").exists() else {},
    )]
    if need:
        print(f"\nTranscribing {len(need)} file(s) with {TRANSCRIBE_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=TRANSCRIBE_WORKERS) as ex:
            futs = {ex.submit(transcribe, AUDIO_DIR / f"{s['id']}.wav", s["id"]): s
                    for s in need}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    fut.result()
                    print(f"  transcribed: {s['title'][:50]}")
                except Exception as e:
                    print(f"  transcription FAILED {s['id']}: {e}")
    else:
        print("\nAll transcriptions already complete.")

    # Phase 3: segment + finalise serially (fast).
    for i, sermon in enumerate(sermons, 1):
        print(f"\n[{i}/{len(sermons)}] {sermon['title']}")
        try:
            wav = download_audio(sermon)
            transcription = transcribe(wav, sermon["id"])
            segments = segment(transcription, sermon["id"])
            fr = transcription.get("final_results", [])
            sermon["status"] = "done" if _coverage_ok(wav, transcription) else "incomplete"
            sermon["num_finals"] = len(fr)
            sermon["num_segments"] = len(segments)
            sermon["audio_end"] = round(fr[-1]["end"], 1) if fr else 0
        except Exception as e:
            sermon["status"] = f"error: {e}"
            print(f"  FAILED: {e}")
        save_manifest(manifest)

    done = sum(1 for s in sermons if s["status"] == "done")
    print(f"\nDone: {done}/{len(sermons)} sermons processed.")


if __name__ == "__main__":
    main()
