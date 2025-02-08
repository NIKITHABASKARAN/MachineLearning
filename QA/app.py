import os
from flask import Flask, render_template, request, send_file, flash, redirect
import pdfplumber
import docx
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}
app.secret_key = 'supersecretkey'

# Hugging Face models for question and answer generation
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")
answer_generator = pipeline("question-answering", model="deepset/roberta-base-squad2")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            return ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif ext == 'docx':
        doc = docx.Document(file_path)
        return ' '.join(para.text for para in doc.paragraphs)
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

def generate_short_answer_questions(input_text, num_questions):
    
    sentences = input_text.split('. ')
    questions = []
    sentence_index = 0
    
    
    while len(questions) < num_questions and sentence_index < len(sentences):
        sentence = sentences[sentence_index]
        
        if len(sentence.strip()) > 10:  
            try:
                # Generate a question using T5 model
                question = question_generator(f"generate question: {sentence}")[0]['generated_text']
                
                # Generate answer using RoBERTa model
                
                start_idx = max(0, sentence_index - 1)
                end_idx = min(len(sentences), sentence_index + 2)
                context = '. '.join(sentences[start_idx:end_idx])
                
                answer = answer_generator(
                    question=question,
                    context=context
                )
                
                # Only if a reasonable confidence score
                if answer['score'] > 0.1:
                    questions.append({
                        "question": question,
                        "answer": answer['answer']
                    })
            except Exception as e:
                print(f"Error generating Q&A for sentence: {e}")
                
        sentence_index += 1
        
       
        # go through the sentences again to generate more questions
        if sentence_index >= len(sentences) and len(questions) < num_questions:
            sentence_index = 0
           
            for i in range(len(sentences)):
                if len(questions) >= num_questions:
                    break
                    
                start_idx = max(0, i - 2)
                end_idx = min(len(sentences), i + 3)
                context = '. '.join(sentences[start_idx:end_idx])
                
                try:
                    question = question_generator(f"generate question: {context}")[0]['generated_text']
                    answer = answer_generator(
                        question=question,
                        context=context
                    )
                    
                    if answer['score'] > 0.1 and not any(q['question'] == question for q in questions):
                        questions.append({
                            "question": question,
                            "answer": answer['answer']
                        })
                except Exception as e:
                    print(f"Error in second pass: {e}")
                    continue
    
    
    return questions[:num_questions]

def save_to_file(content, filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'file' not in request.files:
        flash("No file uploaded!", "error")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash("No file selected!", "error")
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        text = extract_text_from_file(filepath)
        if not text:
            flash("Failed to extract text from the file.", "error")
            return redirect(request.url)
        
        try:
            num_questions = int(request.form.get('num_questions', 5))
        except ValueError:
            flash("Invalid number of questions!", "error")
            return redirect(request.url)
        
        questions = generate_short_answer_questions(text, num_questions)
        
        combined_content = ""
        for i, qa in enumerate(questions, 1):
            combined_content += f"Question {i}: {qa['question']}\nAnswer: {qa['answer']}\n\n"
        
        txt_filename = f"short_answer_questions_{filename}.txt"
        save_to_file(combined_content, txt_filename)
        
        return render_template('results.html', questions=questions, txt_filename=txt_filename)
    
    flash("Invalid file format! Allowed formats: PDF, TXT, DOCX.", "error")
    return redirect(request.url)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    app.run(debug=True)