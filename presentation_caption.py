"""Live captions for presentations: transcribes the microphone with Whisper, locally, and prints each sentence
when you pause.

Run:   python3 presentation_caption.py
Quit:  Ctrl+C.
"""

from __future__ import annotations

import argparse
import collections
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pywhispercpp.model import Model, Segment

Audio = npt.NDArray[np.float32]

RATE = 16000
FRAME = RATE // 50  # 20 ms, the unit of the voice detection
# Whisper's known hallucinations on silence or noise: web addresses, video-outro phrases and sound tags
HALLUCINATION = re.compile(
    r"www\.|https?:|\.(com|org|net)\b|thank(s| you) for watching|subscribe|subtitles|amara\.org|captions by|"
    r"^\s*[\[(].*[\])]\s*$",
    re.I,
)


def load_model(name: str, threads: int, models_dir: str | None) -> Model:
    """The Whisper model, downloaded on first use."""
    from pywhispercpp.model import Model

    return Model(
        name,
        models_dir=models_dir,
        redirect_whispercpp_logs_to=None,
        n_threads=threads,
        print_progress=False,
        print_realtime=False,
    )


def to_audio(pcm: bytes) -> Audio:
    """16-bit PCM bytes as floats in [-1, 1]."""
    return (np.frombuffer(pcm, np.int16) / 32768).astype(np.float32)


def read_mic(source: str | None, chunks: queue.Queue[Audio]) -> None:
    """Puts 100 ms chunks of 16 kHz mono audio from PulseAudio or PipeWire into `chunks`."""
    cmd = ["parec", "--raw", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=50"]
    if source:
        cmd.append(f"--device={source}")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        assert p.stdout is not None
        while chunk := p.stdout.read(RATE // 10 * 2):
            chunks.put(to_audio(chunk))


def read_file(path: str, chunks: queue.Queue[Audio]) -> None:
    """Plays a 16 kHz mono 16-bit WAV file into `chunks` at real time, for testing without a microphone."""
    with wave.open(path) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            sys.exit(f"{path}: needs 16 kHz mono 16-bit audio")
        while frames := w.readframes(RATE // 10):
            chunks.put(to_audio(frames))
            time.sleep(0.1)
    for _ in range(20):  # a pause, so the last sentence ends
        chunks.put(np.zeros(RATE // 10, np.float32))
        time.sleep(0.1)


class Captioner:
    """Turns the audio stream into sentences (`done`) and the sentence being spoken (`partial`)."""

    def __init__(self, model: Model, language: str) -> None:
        self.model = model
        self.language = language
        # Loudness of the last 10 s of 20 ms frames, to tell speech from the room's noise
        self.levels: collections.deque[float] = collections.deque(maxlen=50 * 10)
        self.done: list[str] = []
        self.partial = ""
        self.lock = threading.Lock()

    def text(self) -> str:
        """The last sentences and the one being spoken."""
        with self.lock:
            return " ".join(self.done[-3:] + [self.partial]).strip()

    def run(self, chunks: queue.Queue[Audio]) -> None:
        """Transcribes the sentence being spoken whenever new audio came in, until a pause ends it."""
        buf = np.zeros(0, np.float32)
        while True:
            new = [chunks.get()]
            while not chunks.empty():
                new.append(chunks.get())
            buf = np.concatenate([buf, *new])
            loud = np.flatnonzero(self._speech(buf, sum(len(c) for c in new)))
            if not len(loud):
                buf = buf[-RATE // 2 :]  # keep the start of a word that may be coming
                continue
            if loud[0] * FRAME > RATE // 2:  # drop the silence before the sentence
                buf = buf[loud[0] * FRAME - RATE // 4 :]
                continue
            end = (loud[-1] + 1) * FRAME
            if len(loud) * FRAME < 0.3 * RATE:  # a click or a breath, not speech yet
                if len(buf) - end > 0.6 * RATE:
                    buf = buf[end:]
                continue
            segments = self._transcribe(buf)
            if len(buf) - end > 0.6 * RATE:  # a pause: the sentence is over
                self._commit(segments)
                buf = buf[end:]
            elif len(buf) > 12 * RATE and len(segments) > 1:  # a long sentence: keep only its last part open
                self._commit(segments[:-1], self._join(segments[-1:]))
                buf = buf[segments[-1].t0 * RATE // 100 :]  # t0 is in centiseconds
            elif len(buf) > 25 * RATE:
                self._commit(segments)
                buf = np.zeros(0, np.float32)
            else:
                with self.lock:
                    self.partial = self._join(segments)

    def _speech(self, audio: Audio, new: int) -> npt.NDArray[np.bool_]:
        """Which 20 ms frames of `audio` are speech: louder than -50 dBFS and than four times the room's noise, the
        quietest fifth of the recent frames. `new` samples at its end were not seen before."""
        n = len(audio) // FRAME
        rms = np.sqrt(np.mean(audio[: n * FRAME].reshape(n, FRAME) ** 2, axis=1))
        self.levels.extend(rms[-(new // FRAME) :].tolist() if new >= FRAME else [])
        noise = float(np.percentile(self.levels, 20)) if self.levels else 0.0
        speech: npt.NDArray[np.bool_] = rms > max(4 * noise, 0.003)
        return speech

    def _transcribe(self, audio: Audio) -> list[Segment]:
        segments = self.model.transcribe(audio, language=self.language, no_context=True, suppress_blank=True)
        return [s for s in segments if s.text.strip() and not HALLUCINATION.search(s.text)]

    def _join(self, segments: list[Segment]) -> str:
        return " ".join(s.text.strip() for s in segments).strip()

    def _commit(self, segments: list[Segment], partial: str = "") -> None:
        text = self._join(segments)
        with self.lock:
            if text:
                self.done.append(text)
                print(text, flush=True)
            self.partial = partial


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--model", default="base.en", help="Whisper model: tiny.en, base.en, small.en, ... (default %(default)s)"
    )
    parser.add_argument(
        "--language", default="en", help="spoken language (default %(default)s); models ending in .en are English only"
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4), help="CPU threads for Whisper")
    parser.add_argument("--models-dir", help="where Whisper models are downloaded (default: the user data folder)")
    parser.add_argument(
        "--source", help="PulseAudio source, see `pactl list short sources` (default: the default microphone)"
    )
    parser.add_argument("--file", help="a 16 kHz mono WAV file instead of the microphone, for testing")
    args = parser.parse_args()

    model = load_model(args.model, args.threads, args.models_dir)
    captioner = Captioner(model, args.language)
    print(f"Captions with Whisper {args.model}. Speak.", file=sys.stderr)
    chunks: queue.Queue[Audio] = queue.Queue()
    if args.file:
        reader = threading.Thread(target=read_file, args=(args.file, chunks), daemon=True)
    else:
        reader = threading.Thread(target=read_mic, args=(args.source, chunks), daemon=True)
    reader.start()
    threading.Thread(target=captioner.run, args=(chunks,), daemon=True).start()
    try:
        reader.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
