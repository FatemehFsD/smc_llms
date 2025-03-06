import os
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

PDF_PATH = "in/om_hrs050_en-v_1_388aa0ae.pdf"

# Load models once
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def extract_text_with_page_numbers(pdf_path):
    doc = fitz.open(pdf_path)
    pdf_text = {}
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pdf_text[page_num + 1] = text
    return pdf_text

def create_embeddings(model, pdf_text):
    embeddings = {}
    for page_num, text in pdf_text.items():
        sentences = text.split('\n')
        embeddings[page_num] = model.encode(sentences, convert_to_tensor=True)
    return embeddings

def search_answer_in_pdf(question, pdf_text, pdf_embeddings, model):
    question_embedding = model.encode(question, convert_to_tensor=True)
    best_score = -1
    best_page = None
    best_sentence = None

    for page_num, embeddings in pdf_embeddings.items():
        scores = util.pytorch_cos_sim(question_embedding, embeddings)
        max_score, max_idx = scores.max(dim=1)
        if max_score.item() > best_score:
            best_score = max_score.item()
            best_page = page_num
            best_sentence = pdf_text[page_num].split('\n')[max_idx.item()]

    if best_score > 0.5:
        return f"Answer found on page {best_page}: {best_sentence}"
    else:
        return "No answer found in the PDF."

def generate_answer(question, context):
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = gpt2_tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1)

    answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

def run_model(question):
    pdf_text = extract_text_with_page_numbers(PDF_PATH)
    pdf_embeddings = create_embeddings(sentence_model, pdf_text)
    retrieved_text = search_answer_in_pdf(question, pdf_text, pdf_embeddings, sentence_model)
    generated_answer = generate_answer(question, retrieved_text)
    
    return generated_answer

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = run_model(question)
    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
