FROM python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY configs/rest-api-requirements.txt /tmp/rest-api-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/rest-api-requirements.txt

WORKDIR /app
EXPOSE 5005
CMD ["python", "/app/app.py"]
