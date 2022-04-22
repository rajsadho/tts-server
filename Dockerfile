FROM rajivsadho/taco2-trini-f:1

WORKDIR /app

RUN pip install TTS

COPY . .

CMD ["tts-server", "--vocoder_path", "./model_data/vocoders/model_file.pth.tar", "--vocoder_config_path", "./model_data/vocoders/config.json", "--model_path", "./model_data/models/female.pth", "--config_path", "./model_data/models/config.json"]