FROM rajivsadho/pytorch-tts

WORKDIR /app

COPY . .

CMD ["python", "app.py"]