# 1. الصورة الأساسية لنظام بايثون
FROM python:3.10-slim

# 2. تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# 3. نسخ ملف المتطلبات وتثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. نسخ كامل ملفات المشروع
COPY . .

# 5. كشف المنفذ الافتراضي لمنصة Render
EXPOSE 10000

# 6. أمر تشغيل خادم Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
