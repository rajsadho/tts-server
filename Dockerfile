FROM rajivsadho/male-tts

WORKDIR /app

COPY . .

CMD ["python", "app.py"]
