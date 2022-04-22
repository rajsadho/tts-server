FROM rajivsadho/taco2-trini-f:1

WORKDIR /app

RUN pip install TTS

COPY . .

CMD ["python", "app.py"]