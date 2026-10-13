FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# تثبيت المتطلبات
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# نسخ ملفات المشروع
COPY . .

# تجهيز مجلد قاعدة المعرفة
RUN mkdir -p /app/data/chroma_db

# منفذ الخدمة
EXPOSE 10000

# تشغيل FastAPI على Render
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
