FROM rajivsadho/pytorch-tts

WORKDIR /app

ARG use_cuda=False
ENV use_cuda=${use_cuda}

ARG config_path=False
ENV config_path=${config_path}

ARG model_path=False
ENV model_path=${model_path}

ARG vocoder_config_path=False
ENV vocoder_config_path=${vocoder_config_path}

ARG vocoder_path=False
ENV vocoder_path=${vocoder_path}

ARG debug=False
ENV debug=${debug}


COPY . .

CMD ["python", "app.py"]