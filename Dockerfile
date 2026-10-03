# 1. الصورة الأساسية لنظام بايثون
FROM python:3.10-slim

# 2. تثبيت أدوات النظام الأساسية بالإضافة إلى Node.js و npm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 3. مسار العمل داخل الحاوية
WORKDIR /app

# 4. تحديث أداة pip
RUN pip install --no-cache-dir --upgrade pip

# 5. نسخ متطلبات بايثون وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. نسخ باقي ملفات التطبيق (بما فيها ملفات الواجهة وملفات npm)
COPY . .

# 7. تثبيت حزم npm (إذا وجد ملف package.json في المشروع)
RUN if [ -f package.json ]; then npm install; fi

# 8. منفذ Render الافتراضي
EXPOSE 10000

# 9. أمر تشغيل التطبيق
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
