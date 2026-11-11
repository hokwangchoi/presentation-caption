"""Live captions for presentations: transcribes the microphone with Whisper, locally, and shows the text on top of
every window, so sharing the whole screen shares the captions too.

Run:   python3 presentation_caption.py [--vocab words.txt]
Hover over the captions to show their controls: colors, font size, finish, close, and a resize corner. Drag them to
move.
Finish saves the transcript of the talk to transcript-<date>-<time>.txt in the current folder and quits.
Quit:  the close button, right-click, or Ctrl+C in the terminal. After 2 s of silence, the captions are cleared.

The transcript is also printed to the terminal, one line per sentence.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import ctypes.util
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import tkinter as tk

    from PIL import Image, ImageFont
    from pywhispercpp.model import Model, Segment

Audio = npt.NDArray[np.float32]

RATE = 16000
FRAME = RATE // 50  # 20 ms, the unit of the voice detection
# Whisper's known hallucinations on silence or noise: web addresses, video-outro phrases, sound tags like [BLANK_AUDIO]
# or (music), even cut short, and punctuation alone
HALLUCINATION = re.compile(
    r"www\.|https?:|\.(com|org|net)\b|thank(s| you) for watching|subscribe|subtitles|amara\.org|captions by|"
    r"^\s*[\[(]|^\W*$",
    re.I,
)
# Sound tags and brackets left inside a sentence
TAGS = re.compile(r"\s*[\[(][^\])]*[\])]?|[\[\]()]")
# Color schemes, in the order of the colors button: text, its outline and the background (None: transparent)
SCHEMES: dict[str, tuple[str, str | None, str | None]] = {
    "black-outlined": ("black", "white", None),
    "white-outlined": ("white", "black", None),
    "black-on-white": ("black", None, "white"),
    "white-on-black": ("white", None, "black"),
}


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


def load_vocab(path: str | None) -> list[str]:
    """The terms of a vocabulary file: one per line, `#` starts a comment."""
    if not path:
        return []
    with open(path) as f:
        return [term for line in f if (term := line.split("#")[0].strip())]


def to_audio(pcm: bytes) -> Audio:
    """16-bit PCM bytes as floats in [-1, 1]."""
    return (np.frombuffer(pcm, np.int16) / 32768).astype(np.float32)


class Microphone:
    """Records 16 kHz mono audio from PulseAudio or PipeWire with parec."""

    def __init__(self, source: str | None) -> None:
        cmd = ["parec", "--raw", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=50"]
        if source:
            cmd.append(f"--device={source}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self, chunks: queue.Queue[Audio]) -> None:
        """Puts 100 ms chunks of audio into `chunks` until stopped."""
        assert self.process.stdout is not None
        while chunk := self.process.stdout.read(RATE // 10 * 2):
            chunks.put(to_audio(chunk))

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait()


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

    def __init__(self, model: Model, language: str, vocab: list[str]) -> None:
        self.model = model
        self.language = language
        # Whisper spells the terms of its prompt right more often, and the rest are fixed after (Numpy -> NumPy)
        self.prompt = ", ".join(vocab) + "." if vocab else ""
        self.fixes = [(re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.I), t) for t in vocab if t != t.lower()]
        # Loudness of the last 10 s of 20 ms frames, to tell speech from the room's noise
        self.levels: collections.deque[float] = collections.deque(maxlen=50 * 10)
        self.done: list[str] = []
        self.partial = ""
        self.shown = 0  # the first sentence on screen: the earlier ones are cleared after a silence
        self.lock = threading.Lock()
        # Whisper encodes 30 s windows. The whole window is the most accurate, and fast on a GPU; where it is slow, as
        # on most CPUs, only the audio there is gets encoded, many times faster for the few seconds of a sentence
        self.full_window = self._time_full_window() < 0.15

    def text(self) -> str:
        """The last sentences on screen and the one being spoken."""
        with self.lock:
            return " ".join(self.done[self.shown :][-3:] + [self.partial]).strip()

    def transcript(self) -> str:
        """All the sentences so far, one per line, with the one being spoken."""
        with self.lock:
            return "\n".join(self.done + ([self.partial] if self.partial else [])) + "\n"

    def run(self, chunks: queue.Queue[Audio], stop: threading.Event) -> None:
        """Transcribes the sentence being spoken every quarter second of new audio, until a pause ends it, and clears
        the screen after 2 s of silence. Returns once `stop` is set and the transcription under way is over."""
        buf = np.zeros(0, np.float32)
        quiet = 0  # samples since the last speech
        while not stop.is_set():
            new: list[Audio] = []
            while sum(len(c) for c in new) < RATE // 4 and not stop.is_set():
                try:
                    new.append(chunks.get(timeout=0.1))
                except queue.Empty:
                    pass
            while not chunks.empty():
                new.append(chunks.get())
            if not new:
                continue
            added = sum(len(c) for c in new)
            buf = np.concatenate([buf, *new])
            loud = np.flatnonzero(self._speech(buf, added))
            quiet = len(buf) - (loud[-1] + 1) * FRAME if len(loud) else quiet + added
            if quiet > 2 * RATE:  # so that the last words do not mix with the next ones
                with self.lock:
                    if not self.partial:
                        self.shown = len(self.done)
            if not len(loud):
                buf = buf[-RATE // 2 :]  # keep the start of a word that may be coming
                continue
            if loud[0] * FRAME > RATE // 2:  # drop the silence before the sentence
                buf = buf[loud[0] * FRAME - RATE // 4 :]
                continue
            end = (loud[-1] + 1) * FRAME
            if len(loud) * FRAME < 0.2 * RATE:  # a click or a breath, not speech yet
                if len(buf) - end > 0.6 * RATE:
                    buf = buf[end:]
                continue
            segments = self._transcribe(buf)
            if len(buf) - end > 0.6 * RATE:  # a pause: the sentence is over
                self._commit(segments)
                buf = buf[end:]
            elif len(buf) > 8 * RATE and len(segments) > 1:  # a long sentence: keep only its last part open
                self._commit(segments[:-1], self._join(segments[-1:]))
                buf = buf[segments[-1].t0 * RATE // 100 :]  # t0 is in centiseconds
            elif len(buf) > 25 * RATE:
                self._commit(segments)
                buf = np.zeros(0, np.float32)
            elif len(text := self._join(segments)) < 20 * len(buf) // RATE + 20:  # longer is Whisper repeating itself
                with self.lock:
                    self.partial = text

    def _speech(self, audio: Audio, new: int) -> npt.NDArray[np.bool_]:
        """Which 20 ms frames of `audio` are speech: louder than three times the room's noise, the quietest tenth of
        the last 10 s, and than -50 dBFS. The threshold stays below -30 dBFS, so that talking from the start, before
        any silence, is not taken for noise. `new` samples at its end were not seen before."""
        n = len(audio) // FRAME
        rms = np.sqrt(np.mean(audio[: n * FRAME].reshape(n, FRAME) ** 2, axis=1))
        self.levels.extend(rms[-(new // FRAME) :].tolist() if new >= FRAME else [])
        noise = float(np.percentile(self.levels, 10)) if len(self.levels) >= 50 else 0.0
        speech: npt.NDArray[np.bool_] = rms > min(max(3 * noise, 0.003), 0.03)
        return speech

    def _time_full_window(self) -> float:
        """Seconds to transcribe 1 s of silence with the whole window, after a first run that loads the model."""
        silence = np.zeros(RATE, np.float32)
        for _ in range(2):
            start = time.monotonic()
            self.model.transcribe(silence, language=self.language, no_context=True, audio_ctx=0)
        return time.monotonic() - start

    def _transcribe(self, audio: Audio) -> list[Segment]:
        # 50 positions of the window per second of audio, with a margin
        context = 0 if self.full_window else min(1500, len(audio) * 50 // RATE + 64)
        segments = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=self.prompt,
            no_context=True,
            suppress_blank=True,
            audio_ctx=context,
            temperature_inc=0.0,  # no slow retries at higher temperatures: the next update corrects a bad guess
            max_tokens=len(audio) * 6 // RATE + 16,  # about twice the words of fast speech, to stop repetition loops
        )
        return [s for s in segments if s.text.strip() and not HALLUCINATION.search(s.text)]

    def _join(self, segments: list[Segment]) -> str:
        text = TAGS.sub("", " ".join(s.text.strip() for s in segments)).strip()
        for pattern, term in self.fixes:
            text = pattern.sub(term, text)
        return text

    def _commit(self, segments: list[Segment], partial: str = "") -> None:
        text = self._join(segments)
        with self.lock:
            if text:
                self.done.append(text)
                print(text, flush=True)
            self.partial = partial


class Shape:
    """Cuts an X11 window to a mask with the X Shape extension, so only the drawn pixels show and get the mouse.
    Unlike a transparent background, it needs no compositor."""

    def __init__(self) -> None:
        x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        xext = ctypes.CDLL(ctypes.util.find_library("Xext") or "libXext.so.6")
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XCreateBitmapFromData.restype = ctypes.c_ulong
        x11.XCreateBitmapFromData.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        x11.XFreePixmap.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XFlush.argtypes = [ctypes.c_void_p]
        x11.XFree.argtypes = [ctypes.c_void_p]
        x11.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, *[ctypes.c_void_p] * 4]
        xext.XShapeCombineMask.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_int,
        ]
        self.x11 = x11
        self.xext = xext
        self.display = x11.XOpenDisplay(None)

    def top(self, window: int) -> int:
        """The ancestor of `window` that is a child of the root window, the one X shows: Tk wraps its windows."""
        root, parent = ctypes.c_ulong(), ctypes.c_ulong()
        children, count = ctypes.c_void_p(), ctypes.c_uint()
        while True:
            refs = (ctypes.byref(root), ctypes.byref(parent), ctypes.byref(children), ctypes.byref(count))
            self.x11.XQueryTree(self.display, window, *refs)
            if children:
                self.x11.XFree(children)
            if parent.value in (0, root.value):
                return window
            window = parent.value

    def set(self, window: int, mask: npt.NDArray[np.bool_] | None) -> None:
        """Shows only the pixels of `window` where `mask` is true, or all of them if it is None."""
        window = self.top(window)
        bounding, set_op = 0, 0  # ShapeBounding, ShapeSet
        if mask is None:
            self.xext.XShapeCombineMask(self.display, window, bounding, 0, 0, 0, set_op)
        else:
            height, width = mask.shape
            bits = np.packbits(mask, axis=1, bitorder="little").tobytes()  # rows of X bitmap data
            pixmap = self.x11.XCreateBitmapFromData(self.display, window, bits, width, height)
            self.xext.XShapeCombineMask(self.display, window, bounding, 0, 0, pixmap, set_op)
            self.x11.XFreePixmap(self.display, pixmap)
        self.x11.XFlush(self.display)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A bold sans-serif font, from fontconfig if available."""
    from PIL import ImageFont

    if shutil.which("fc-match"):
        path = subprocess.run(["fc-match", "-f", "%{file}", "sans:bold"], capture_output=True, text=True).stdout
        if path:
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def monitor_at(x: int, y: int) -> tuple[int, int, int, int] | None:
    """`(x, y, width, height)` of the monitor showing the point, from xrandr, or None."""
    try:
        out = subprocess.run(["xrandr", "--listmonitors"], capture_output=True, text=True).stdout
    except OSError:
        return None
    for m in re.finditer(r"(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)", out):
        w, h, mx, my = map(int, m.groups())
        if mx <= x < mx + w and my <= y < my + h:
            return mx, my, w, h
    return None


class CaptionWindow:
    """The last lines of captions that fit, across the monitor under the mouse, on top of every window. The letters
    are drawn with Pillow and the window is cut to their shape. Hovering shows a solid bar with the controls."""

    PAD = 16
    GRIP = 20

    def __init__(self, captioner: Captioner, font_size: int, lines: int, position: str, colors: str) -> None:
        import tkinter as tk

        self.captioner = captioner
        self.scheme = colors
        self.font_size = font_size
        self.font = load_font(font_size)
        self.text = ""
        self.hovered = False
        self.drag_offset: tuple[int, int] | None = None
        self.shape = Shape()
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # not framed or tiled by the window manager
        self.root.attributes("-topmost", True)
        self.label = tk.Label(self.root, borderwidth=0, highlightthickness=0)
        self.label.pack(fill="both", expand=True)
        self.buttons = [
            self._button("Colors", self.next_scheme),
            self._button("A-", lambda: self.zoom(-4)),
            self._button("A+", lambda: self.zoom(4)),
            self._button("Finish", self.finish),
            self._button("Close", self.close),
        ]
        self.grip = tk.Frame(self.root, width=self.GRIP, height=self.GRIP, bg="darkgray", cursor="bottom_right_corner")
        screen = (0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        mx, my, width, mh = monitor_at(*self.root.winfo_pointerxy()) or screen
        height = lines * self.line_height() + 2 * self.PAD
        y = my + 60 if position == "top" else my + mh - height - 60
        self.root.geometry(f"{width}x{height}+{mx}+{y}")
        self.size = (0, 0)  # so that the first refresh draws the window, and cuts it to the handle
        self.image: object = None  # the PhotoImage shown, kept from garbage collection
        self.label.bind("<Enter>", self.enter)
        self.root.bind("<Leave>", self.leave)
        self.label.bind("<Button-1>", self.press)
        self.label.bind("<B1-Motion>", self.move)
        self.label.bind("<ButtonRelease-1>", self.release)
        self.grip.bind("<B1-Motion>", self.resize)
        self.label.bind("<Button-4>", lambda e: self.zoom(4))  # scroll up and down
        self.label.bind("<Button-5>", lambda e: self.zoom(-4))
        self.label.bind("<Button-3>", lambda e: self.close())
        self.root.after(50, self.refresh)

    def _button(self, label: str, action: Callable[[], object]) -> tk.Button:
        import tkinter as tk

        return tk.Button(self.root, text=label, command=action, padx=6, pady=0, cursor="hand2")

    def run(self) -> None:
        """Shows the captions until closed."""
        signal.signal(signal.SIGINT, lambda *_: self.close())  # Ctrl+C, handled when `refresh` runs
        self.root.mainloop()
        self.root.destroy()

    def close(self) -> None:
        """Ends `run`. The window is destroyed after its callbacks, so that none runs on a destroyed window."""
        self.root.quit()

    def line_height(self) -> int:
        return int(self.font.getbbox("Ag")[3] * 1.2)

    def refresh(self) -> None:
        text = self.captioner.text()
        size = (self.root.winfo_width(), self.root.winfo_height())
        if text != self.text or size != self.size:
            self.text, self.size = text, size
            self.redraw()
        self.root.lift()
        self.root.after(100, self.refresh)

    def next_scheme(self) -> None:
        names = list(SCHEMES)
        self.scheme = names[(names.index(self.scheme) + 1) % len(names)]
        print(f"Colors: {self.scheme}", file=sys.stderr)
        self.redraw()

    def finish(self) -> None:
        """Saves the transcript of the talk in the current folder and quits."""
        path = time.strftime("transcript-%Y%m%d-%H%M%S.txt")
        with open(path, "w") as f:
            f.write(self.captioner.transcript())
        print(f"Transcript saved to {os.path.abspath(path)}", file=sys.stderr)
        self.close()

    def zoom(self, step: int) -> None:
        self.font_size = max(16, min(160, self.font_size + step))
        self.font = load_font(self.font_size)
        self.redraw()

    def lines(self) -> list[str]:
        """The last lines of the text that fit, wrapped to the width left of the buttons, so that showing them
        does not cover the text."""
        buttons = sum(b.winfo_reqwidth() + 6 for b in self.buttons) + self.GRIP + 8
        width = self.size[0] - self.PAD - buttons
        fit = max(1, (self.size[1] - 2 * self.PAD) // self.line_height())
        out: list[str] = []
        line = ""
        for word in self.text.split():
            if line and self.font.getlength(line + " " + word) > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        return (out + [line])[-fit:]

    def draw(self) -> Image.Image:
        """The captions as an image, transparent where the window is cut away."""
        from PIL import Image, ImageDraw

        fg, outline, bg = SCHEMES[self.scheme]
        width, height = self.size
        back = "gray" if self.hovered else bg
        image = Image.new("RGBA", (width, height), back or (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        lines = self.lines()
        stroke = max(2, self.font_size // 14) if outline else 0
        for i, line in enumerate(lines):
            y = height - self.PAD - (len(lines) - i) * self.line_height()
            draw.text((self.PAD, y), line, font=self.font, fill=fg, stroke_width=stroke, stroke_fill=outline)
        if not self.text and not back:  # a handle to hover and grab, as the window is invisible without text
            draw.rectangle((0, height - 12, 12, height), fill="gray")
        return image

    def redraw(self) -> None:
        from PIL import ImageTk

        image = self.draw()
        photo = ImageTk.PhotoImage(image)
        self.label.configure(image=photo)
        self.image = photo
        self.root.update_idletasks()
        window = int(self.root.wm_frame(), 16)
        if self.hovered or SCHEMES[self.scheme][2]:
            self.shape.set(window, None)
        else:
            self.shape.set(window, np.asarray(image.getchannel("A")) > 127)

    def enter(self, event: tk.Event[tk.Misc]) -> None:
        if self.hovered:
            return
        self.hovered = True
        x = self.size[0] - self.GRIP - 8
        for button in reversed(self.buttons):
            x -= button.winfo_reqwidth() + 6
            button.place(x=x, y=6)
        self.grip.place(relx=1.0, rely=1.0, anchor="se")
        self.redraw()

    def leave(self, event: tk.Event[tk.Misc]) -> None:
        px, py = self.root.winfo_pointerxy()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        if self.drag_offset or (x <= px < x + self.size[0] and y <= py < y + self.size[1]):
            return  # onto a button, or dragging
        self.hovered = False
        for widget in [*self.buttons, self.grip]:
            widget.place_forget()
        self.redraw()

    def press(self, event: tk.Event[tk.Misc]) -> None:
        self.drag_offset = (event.x, event.y)

    def move(self, event: tk.Event[tk.Misc]) -> None:
        if self.drag_offset:
            x = self.root.winfo_x() + event.x - self.drag_offset[0]
            y = self.root.winfo_y() + event.y - self.drag_offset[1]
            self.root.geometry(f"+{x}+{y}")

    def release(self, event: tk.Event[tk.Misc]) -> None:
        self.drag_offset = None

    def resize(self, event: tk.Event[tk.Misc]) -> None:
        width = max(300, event.x_root - self.root.winfo_rootx())
        height = max(self.line_height() + 2 * self.PAD, event.y_root - self.root.winfo_rooty())
        self.root.geometry(f"{width}x{height}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--model", default="base.en", help="Whisper model: tiny.en, base.en, small.en, ... (default %(default)s)"
    )
    parser.add_argument(
        "--language", default="en", help="spoken language (default %(default)s); models ending in .en are English only"
    )
    parser.add_argument("--vocab", help="file of terms of the talk, one per line, to spell them right")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4), help="CPU threads for Whisper")
    parser.add_argument("--models-dir", help="where Whisper models are downloaded (default: the user data folder)")
    parser.add_argument(
        "--source", help="PulseAudio source, see `pactl list short sources` (default: the default microphone)"
    )
    parser.add_argument("--file", help="a 16 kHz mono WAV file instead of the microphone, for testing")
    parser.add_argument(
        "--colors",
        choices=list(SCHEMES),
        default="black-outlined",
        help="color scheme at start, also changed with the colors button (default %(default)s)",
    )
    parser.add_argument("--font-size", type=int, default=40, help="in pixels (default %(default)s)")
    parser.add_argument("--lines", type=int, default=2, help="height of the window at start, in lines")
    parser.add_argument("--position", choices=["bottom", "top"], default="bottom")
    parser.add_argument("--no-window", action="store_true", help="only print the transcript")
    args = parser.parse_args()

    model = load_model(args.model, args.threads, args.models_dir)
    captioner = Captioner(model, args.language, load_vocab(args.vocab))
    where = "the GPU and the whole window" if captioner.full_window else "the CPU"
    print(f"Captions with Whisper {args.model} on {where}. Speak.", file=sys.stderr)
    chunks: queue.Queue[Audio] = queue.Queue()
    stop = threading.Event()
    mic = None if args.file else Microphone(args.source)
    if mic:
        reader = threading.Thread(target=mic.read, args=(chunks,), daemon=True)
    else:
        reader = threading.Thread(target=read_file, args=(args.file, chunks), daemon=True)
    worker = threading.Thread(target=captioner.run, args=(chunks, stop))
    reader.start()
    worker.start()
    try:
        if args.no_window:
            reader.join()
            while not chunks.empty():  # the end of the recording
                time.sleep(0.1)
            time.sleep(1)
        else:
            CaptionWindow(captioner, args.font_size, args.lines, args.position, args.colors).run()
    except KeyboardInterrupt:
        pass
    finally:
        # Exiting during a transcription aborts the program in CUDA: let the one under way finish
        stop.set()
        if mic:
            mic.stop()
        worker.join()


if __name__ == "__main__":
    main()
