"""Run the trained SmolVLA episode from a live Speechmatics voice command.

The only actionable phrases are ones containing ``plate`` plus either ``cup``
or ``table``.  Other vocabulary is deliberately ignored: this is a keyword
trigger, not a natural-language command parser.

Usage:
    python voice_trigger.py

``SPEECHMATICS_API_KEY`` is read from the environment first, then from the
repository-root .env file (which must remain untracked).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Final

import aiohttp
import sounddevice as sd


ROOT: Final = Path(__file__).resolve().parent
SPEECHMATICS_URL: Final = "wss://eu.rt.speechmatics.com/v2"
SAMPLE_RATE: Final = 16_000
CHANNELS: Final = 1
BLOCK_SIZE: Final = 4_096
UTTERANCE_SILENCE_SECONDS: Final = 1.75
TARGET_WORD_PAIRS: Final = (("plate", "cup"), ("plate", "table"))
# These words intentionally have no effect on matching.  In particular, they
# neither launch nor suppress the episode when a target pair is present.
IGNORED_KEYWORDS: Final = frozenset({"drawer", "open", "close", "arm", "a", "b"})

RUNNER_PYTHON: Final = Path(
    r"E:\Hackathons\infra-summit-vla\.venv\Scripts\python.exe"
)
TRAINED_INSTRUCTION: Final = (
    "Open the drawer, close the drawer, pick up the plate, and place the "
    "plate in the center of the table."
)
RUNNER_COMMAND: Final = (
    str(RUNNER_PYTHON),
    "scratch/run_vla.py",
    "--config",
    "configs/smolvla_rollout_cpu.yaml",
    "--seed",
    "0",
    "--instruction",
    TRAINED_INSTRUCTION,
)


def load_dotenv_key(path: Path = ROOT / ".env") -> str | None:
    """Return SPEECHMATICS_API_KEY without adding a dotenv dependency."""
    key = os.environ.get("SPEECHMATICS_API_KEY")
    if key:
        return key
    if not path.is_file():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != "SPEECHMATICS_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None


def words(text: str) -> set[str]:
    """Normalize a transcript to whole lower-case word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def matches_episode_trigger(text: str) -> bool:
    """True only for 'plate + cup' or 'plate + table' final transcripts."""
    transcript_words = words(text)
    return any(left in transcript_words and right in transcript_words
               for left, right in TARGET_WORD_PAIRS)


async def run_episode() -> bool:
    """Run the one trained instruction and inherit its live terminal output."""
    print("Trigger matched. Launching real SmolVLA runner...", flush=True)
    process = await asyncio.create_subprocess_exec(*RUNNER_COMMAND, cwd=ROOT)
    return_code = await process.wait()
    success = return_code == 0
    print("SUCCESS" if success else "FAILURE", flush=True)
    return success


async def wait_for_recognition_started(ws: aiohttp.ClientWebSocketResponse) -> None:
    """Wait for the service handshake before transmitting microphone audio."""
    while True:
        message = await ws.receive()
        if message.type is aiohttp.WSMsgType.TEXT:
            payload = json.loads(message.data)
            kind = payload.get("message")
            if kind == "RecognitionStarted":
                return
            if kind == "Error":
                raise RuntimeError(f"Speechmatics error: {payload}")
        elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
            raise RuntimeError("Speechmatics closed the connection during startup")
        elif message.type is aiohttp.WSMsgType.ERROR:
            raise RuntimeError(f"Speechmatics WebSocket error: {ws.exception()}")


async def receive_transcripts(
    ws: aiohttp.ClientWebSocketResponse,
    trigger: asyncio.Event,
    *,
    silence_seconds: float = UTTERANCE_SILENCE_SECONDS,
) -> None:
    """Evaluate each final-transcript utterance after a short silence."""
    final_text_buffer: list[str] = []
    loop = asyncio.get_running_loop()
    evaluation_deadline: float | None = None

    def evaluate_utterance() -> bool:
        """Evaluate and always discard one completed utterance."""
        if not final_text_buffer:
            return False

        utterance = " ".join(final_text_buffer)
        final_text_buffer.clear()
        print(f"Evaluating utterance: {utterance}", flush=True)
        if matches_episode_trigger(utterance):
            trigger.set()
            return True
        return False

    while True:
        timeout = (
            None
            if evaluation_deadline is None
            else max(0.0, evaluation_deadline - loop.time())
        )
        try:
            message = await ws.receive(timeout=timeout)
        except asyncio.TimeoutError:
            evaluation_deadline = None
            if evaluate_utterance():
                return
            continue

        if message.type is aiohttp.WSMsgType.TEXT:
            payload = json.loads(message.data)
            kind = payload.get("message")
            if kind == "AddTranscript":
                transcript = payload.get("metadata", {}).get("transcript", "").strip()
                if transcript:
                    print(f"Recognized: {transcript}", flush=True)
                    final_text_buffer.append(transcript)
                    evaluation_deadline = loop.time() + silence_seconds
            elif kind == "Error":
                raise RuntimeError(f"Speechmatics error: {payload}")
        elif message.type is aiohttp.WSMsgType.ERROR:
            raise RuntimeError(f"Speechmatics WebSocket error: {ws.exception()}")
        elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
            evaluate_utterance()
            return


async def stream_microphone(
    ws: aiohttp.ClientWebSocketResponse, trigger: asyncio.Event
) -> None:
    """Send 16 kHz signed-16-bit mono microphone chunks until triggered."""
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)

    def enqueue_audio(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"Microphone status: {status}", file=sys.stderr, flush=True)
        chunk = bytes(indata)

        def put_chunk() -> None:
            if not audio_queue.full():
                audio_queue.put_nowait(chunk)

        loop.call_soon_threadsafe(put_chunk)

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="int16",
        callback=enqueue_audio,
    ):
        print("Listening for 'plate cup' or 'plate table'. Press Ctrl+C to stop.", flush=True)
        while not trigger.is_set():
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.25)
            except TimeoutError:
                continue
            await ws.send_bytes(chunk)


async def check_connection() -> int:
    """Verify Speechmatics authentication without opening audio or MuJoCo."""
    api_key = load_dotenv_key()
    if not api_key:
        print("SPEECHMATICS_API_KEY is missing from the environment or .env.", file=sys.stderr)
        return 2

    url = os.environ.get("SPEECHMATICS_RT_URL", SPEECHMATICS_URL)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(
            url, headers={"Authorization": f"Bearer {api_key}"}, heartbeat=20
        ) as ws:
            await ws.send_json(
                {
                    "message": "StartRecognition",
                    "audio_format": {
                        "type": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": SAMPLE_RATE,
                    },
                    "transcription_config": {"language": "en"},
                }
            )
            await wait_for_recognition_started(ws)
            await ws.send_json({"message": "EndOfStream", "last_seq_no": 0})

    print("Speechmatics authentication and recognition handshake succeeded.")
    return 0


async def main() -> int:
    api_key = load_dotenv_key()
    if not api_key:
        print("SPEECHMATICS_API_KEY is missing from the environment or .env.", file=sys.stderr)
        return 2

    url = os.environ.get("SPEECHMATICS_RT_URL", SPEECHMATICS_URL)
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

    while True:
        trigger = asyncio.Event()
        async with aiohttp.ClientSession(timeout=timeout) as http_session:
            async with http_session.ws_connect(url, headers=headers, heartbeat=20) as ws:
                await ws.send_json(
                    {
                        "message": "StartRecognition",
                        "audio_format": {
                            "type": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": SAMPLE_RATE,
                        },
                        "transcription_config": {
                            "language": "en",
                            "enable_partials": False,
                            "max_delay": 1.0,
                        },
                    }
                )
                await wait_for_recognition_started(ws)

                receiver = asyncio.create_task(receive_transcripts(ws, trigger))
                microphone = asyncio.create_task(stream_microphone(ws, trigger))
                try:
                    done, _ = await asyncio.wait(
                        {receiver, microphone}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()  # Propagate microphone or WebSocket failures.

                    if not trigger.is_set():
                        raise RuntimeError("Speechmatics session ended before a trigger phrase")

                    # Stop the audio producer before the VLA run. This serializes
                    # episodes, so a second transcript cannot launch a duplicate.
                    await microphone
                finally:
                    for task in (receiver, microphone):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(receiver, microphone, return_exceptions=True)
                    if not ws.closed:
                        await ws.send_json({"message": "EndOfStream", "last_seq_no": 0})

        success = await run_episode()
        return 0 if success else 1


if __name__ == "__main__":
    try:
        entrypoint = check_connection() if "--check-connection" in sys.argv else main()
        raise SystemExit(asyncio.run(entrypoint))
    except KeyboardInterrupt:
        print("\nVoice trigger stopped.")
