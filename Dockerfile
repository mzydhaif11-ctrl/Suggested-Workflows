# 1. استخدام إصدار بايثون حديث ومستقر
FROM python:3.10-slim

# 2. تثبيت التحديثات الأساسية لبيئة النظام وتحديث pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# 4. ترقية أداة تثبيت الحزم pip لتفادي مشاكل الحزم الحديثة
RUN pip install --no-cache-dir --upgrade pip

# 5. نسخ ملف المتطلبات أولاً للاستفادة من الـ Cache
COPY requirements.txt .

# 6. تثبيت مكتبات المشروع
RUN pip install --no-cache-dir -r requirements.txt

# 7. نسخ باقي ملفات المشروع إلى داخل الحاوية
COPY . .

# 8. كشف المنفذ الافتراضي لمنصة Render
EXPOSE 10000

# 9. أمر تشغيل خادم FastAPI عبر uvicorn على المنفذ المطلوب
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
