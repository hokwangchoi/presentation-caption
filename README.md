# presentation-caption

Live captions for your talks. [Whisper](https://github.com/openai/whisper) transcribes your microphone locally, and
the captions float over every window with a transparent background, so sharing your screen shares them.

![Captions over a slide](docs/caption-bar.png)

## Install

You need Linux with X11, Python 3.10 or newer with Tk (`sudo apt install python3-tk` on Debian and Ubuntu), and
PulseAudio or PipeWire.

```bash
git clone https://github.com/hokwangchoi/presentation-caption.git
cd presentation-caption
pip install -r requirements.txt
```

It runs on the CPU with [whisper.cpp](https://github.com/ggml-org/whisper.cpp); the first run downloads the model
(140 MB for `base.en`).

## Use

```bash
python3 presentation_caption.py --vocab examples/web-service.txt
```

Share your entire screen, not one window. Hover over the captions for the controls: **Colors**, **A-**, **A+**,
**Finish**, **Close**, and a resize corner. Drag them to move. Before the first words, a small grey square at the
bottom left is the handle.
**Finish** saves the transcript of the talk to `transcript-<date>-<time>.txt` in the folder you started it from.

`--vocab` takes the terms of your talk, one per line, so they come out spelled your way:

| Without | With |
|---|---|
| The pods run behind an **engine X** proxy, and we debug them with **Cubicle**. | The pods run behind an **Nginx** proxy, and we debug them with **kubectl**. |

Other options: `--colors`, `--font-size`, `--position top`, `--source` for another microphone (see
`pactl list short sources`), `--model small.en` for more accuracy on a fast CPU, and `--file talk.wav` to test with a
16 kHz mono recording. See `--help`.

## License

[MIT](LICENSE)
