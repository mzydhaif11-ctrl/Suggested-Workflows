import hashlib
import json
import datetime

class MowjhEngine:
    def __init__(self, project_name="موجة البيان"):
        self.project_name = project_name
        self.registry = []

    def add_innovation_node(self, node_title, author, content):
        """
        إضافة عقدة فكرية جديدة مع بصمة مشفرة لضمان سيادة البيانات والتوثيق
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = f"{node_title}-{author}-{content}-{timestamp}"
        
        # حساب البصمة المشفرة SHA-256 للتوثيق المعتمد
        digital_signature = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        
        node_data = {
            "title": node_title,
            "author": author,
            "timestamp": timestamp,
            "signature": digital_signature,
            "status": "معتمد وموثق سيادياً"
        }
        
        self.registry.append(node_data)
        print(f"✅ تم توثيق الفكرة بنجاح: [{node_title}]")
        return digital_signature

    def export_registry_json(self):
        """تصدير السجل بصيغة بيانات جاهزة للربط مع GitHub"""
        return json.dumps(self.registry, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # تجربة التشغيل العملي لمحرك موجة البيان
    engine = MowjhEngine()
    engine.add_innovation_node(
        node_title="هندسة منصة موجة الذكية",
        author="مزيد الرفعان",
        content="ربط مساحة الأفكار الحية بمستودع التوثيق السيادي"
    )
    print("\n📦 محتوى السجل الموثق جاهز للرفع على المستودع:")
    print(engine.export_registry_json())
