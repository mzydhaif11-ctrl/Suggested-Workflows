# 🚀 دليل النشر والتشغيل (Deployment Guide)

هذا الدليل يوضح خطوات تشغيل **منصة موجة البيان** على بيئتك المحلية، وكذلك كيفية نشرها على منصات السحاب مثل **Render** أو **Google Cloud Run**.

## 1. التشغيل المحلي (Local Development)

### المتطلبات المسبقة:
* Python 3.10+
* حساب في Google AI Studio (للحصول على `GEMINI_API_KEY`)
* حساب أو خادم Qdrant (للحصول على `QDRANT_URL` و `QDRANT_API_KEY`)

### الخطوات:
1. استنسخ المستودع:
   ```bash
   git clone [https://github.com/your-username/Mowjh-Al-Bayan.git](https://github.com/your-username/Mowjh-Al-Bayan.git)
   cd Mowjh-Al-Bayan
