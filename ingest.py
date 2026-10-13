import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# رابط مستودع GitHub الأساسي لمشروعك (لتوليد روابط تفتح الملف مباشرة)
BASE_REPO_URL = "https://github.com/mzydhaif11-ctrl/Suggested-Workflows/blob/main/docs"

# إعداد مقطّع النصوص
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_chunks = []

# البحث عن جميع ملفات الماركداون أو النصوص داخل مجلد docs
doc_files = glob.glob("docs/*.md") + glob.glob("docs/*.txt")

for file_path in doc_files:
    file_name = os.path.basename(file_path)
    # توليد عنوان مقروء من اسم الملف (مثلاً: deployment_guide.md -> Deployment Guide)
    title = file_name.rsplit(".", 1)[0].replace("_", " ").title()
    # رابط الملف المباشر على جيت هب
    file_url = f"{BASE_REPO_URL}/{file_name}"

    loader = TextLoader(file_path, encoding="utf-8")
    raw_docs = loader.load()

    # تقطيع المستند وإضافة البيانات الوصفية لكل مقطع
    chunks = text_splitter.split_documents(raw_docs)
    for chunk in chunks:
        chunk.metadata["source_title"] = title
        chunk.metadata["source_url"] = file_url
        chunk.metadata["file_name"] = file_name

    all_chunks.extend(chunks)

# بعد ذلك يتم حفظ all_chunks في قاعدة بيانات المتجهات (Vector Store)
# vector_db.add_documents(all_chunks)
