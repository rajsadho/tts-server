import io
import json
import os
from typing import Union

from flask import Flask, render_template, request, send_file

from TTS.config import load_config
from TTS.utils.synthesizer import Synthesizer

app = Flask(__name__)

# Hardcoded config. Host does not play well with env.
model_path = "/app/model_data/models/female.pth"
config_path = "/app/model_data/models/config.json"
vocoder_path = "/app/model_data/vocoders/model_file.pth.tar"
vocoder_config_path = "/app/model_data/vocoders/config.json"
port = 5002
use_cuda = False
debug = True
show_details = True

args = {
    "model_path": model_path,
    "config_path": config_path,
    "vocoder_path": vocoder_path,
    "vocoder_config_path": vocoder_config_path,
    "port": port,
    "use_cuda": False,
    "debug": debug,
    "show_details": show_details,
}

# load models
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=use_cuda,
)


use_multi_speaker = (
    hasattr(synthesizer.tts_model, "num_speakers")
    and synthesizer.tts_model.num_speakers > 1
)
speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)
# TODO: set this from SpeakerManager
use_gst = synthesizer.tts_config.get("use_gst", False)


def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=show_details,
        use_multi_speaker=use_multi_speaker,
        speaker_ids=speaker_manager.speaker_ids
        if speaker_manager is not None
        else None,
        use_gst=use_gst,
    )


@app.route("/details")
def details():
    model_config = load_config(config_path)
    if vocoder_config_path is not None and os.path.isfile(vocoder_config_path):
        vocoder_config = load_config(vocoder_config_path)
    else:
        vocoder_config = None

    return render_template(
        "details.html",
        show_details=True,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args,
    )


@app.route("/api/tts", methods=["GET"])
def tts():
    text = request.args.get("text")
    speaker_idx = request.args.get("speaker_id", "")
    style_wav = request.args.get("style_wav", "")
    style_wav = style_wav_uri_to_dict(style_wav)
    print(" > Model input: {}".format(text))
    print(" > Speaker Idx: {}".format(speaker_idx))
    wavs = synthesizer.tts(text, speaker_name=speaker_idx, style_wav=style_wav)
    out = io.BytesIO()
    synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


def main():
    app.run(debug=debug, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
