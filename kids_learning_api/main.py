from fastapi import FastAPI
from pydantic import BaseModel
import re
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ----- تحميل الـ env -----
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# ----- FastAPI -----
app = FastAPI(title="Kids Learning API")

# ----- نموذج بيانات للطلب -----
class LessonRequest(BaseModel):
    lesson_number: int

# ----- تحميل الدروس من الملف مرة واحدة -----
file_path = "Cleaned_DERASAT_full_text.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

pattern = r"(الدرس\s+(?:الأول|الثاني|الثالث).*?)"
parts = re.split(pattern, text)
lessons = [parts[i] + parts[i+1] for i in range(1, len(parts), 2)]
print(f"عدد الدروس: {len(lessons)}")

# ----- دالة توليد قصة -----
def generate_story(selected_lesson: str) -> str:
    prompt = f"""
أنت مساعد تعليمي للأطفال.

المطلوب:
1) اكتب قصة قصيرة لطفل تشرح الدرس التالي بطريقة ممتعة
2) الهدف: جعل المذاكرة ممتعة ومفهومة
3) أسلوب القصة: شيق، تعليم بطريقة القصص
4) طول القصة متوسطة ليست بقصيرة ولا بطويلة
5) قسم القصة إلى فقرات .
4) ضع هذا الفاصل بين كل فقرة والتي بعدها:
---------

مواصفات:
- كل فقرة = مشهد واحد
- الشخصيات الأساسية: Amir and Amira 
- لا تجعل الفقرات طويلة

الدرس:
{selected_lesson}
"""
    response = model.generate_content(prompt)
    return response.text

# ----- دالة توليد MCQ -----
def generate_mcq(paragraph: str):
    prompt = f"""
أنت معلم للأطفال. اقرأ النص التالي واعمل سؤال اختيار من متعدد يدور حول الفكرة الرئيسية للفقرة وليست النقاط الفرعية.

أرجع النتيجة JSON فقط بدون أي كلام إضافي.

النص:
{paragraph}

الصيغة المطلوبة:
{{
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "answer": "A"
}}
"""
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.strip()
        clean_text = clean_text.replace("```json", "").replace("```", "")

        return json.loads(clean_text)

    except Exception as e:
        print("❌ خطأ:", e)
        return None


# ----- Endpoints -----

@app.get("/lessons")
def list_lessons():
    return {"lessons": [lesson.splitlines()[0] for lesson in lessons]}

@app.post("/story")
def story(request: LessonRequest):
    if request.lesson_number < 1 or request.lesson_number > len(lessons):
        return {"error": "رقم الدرس غير صالح"}
    
    selected_lesson = lessons[request.lesson_number - 1]
    story_text = generate_story(selected_lesson)
    return {"story": story_text}

@app.post("/quiz")
def quiz(request: LessonRequest):
    if request.lesson_number < 1 or request.lesson_number > len(lessons):
        return {"error": "رقم الدرس غير صالح"}
    
    selected_lesson = lessons[request.lesson_number - 1]
    story_text = generate_story(selected_lesson)
    story_paragraphs = re.split(r'---------\n', story_text.strip())
    
    mcqs = [generate_mcq(p) for p in story_paragraphs]
    return {"mcqs": mcqs}
