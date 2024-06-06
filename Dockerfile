FROM python:3.10-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python", "app.py"]