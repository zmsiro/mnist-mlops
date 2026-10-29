FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt

# The actual trained Assignment 1 model.pt must be present in the build context.
COPY model.py app.py model.pt ./

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
