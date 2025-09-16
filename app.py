
# =========================
# Imports
# =========================
import os
import torch
import gradio as gr
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import docx
import pandas as pd

# HuggingFace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoProcessor,
    pipeline,
)

# LangChain for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# =========================
# Models
# =========================
QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading Qwen VL model:", QWEN_MODEL)
try:
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    qwen_model = AutoModelForVision2Seq.from_pretrained(
        QWEN_MODEL,
        device_map="cpu",  # Force CPU for Docker compatibility
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).eval()
except Exception as e:
    print(f"⚠️ Error loading Qwen model: {e}")
    raise

# =========================
# RAG Setup (Knowledge Base)
# =========================
KB_PDF_PATH = "./knowledge_base.pdf"
CHROMA_DIR = "./chroma_db"

# Ensure the Chroma directory exists with proper permissions
os.makedirs(CHROMA_DIR, exist_ok=True)
os.chmod(CHROMA_DIR, 0o777)  # Give full permissions

vectorstore = None
if os.path.exists(KB_PDF_PATH):
    print(f"🔎 Loading knowledge base: {KB_PDF_PATH}")
    try:
        # Configure Chroma settings
        from chromadb.config import Settings
        chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_DIR,
            anonymized_telemetry=False
        )

        # Load and process the PDF
        loader = PyPDFLoader(KB_PDF_PATH)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Chroma with settings
        vectorstore = Chroma.from_documents(
            docs,
            embedding_model,
            persist_directory=CHROMA_DIR,
            client_settings=chroma_settings,
        )
    except Exception as e:
        print(f"⚠️ Error loading knowledge base: {e}")
        vectorstore = None
else:
    print("⚠️ No knowledge base PDF found at", KB_PDF_PATH)

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve most relevant chunks from knowledge base."""
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return "\n\n".join([r.page_content for r in results])
    except Exception as e:
        return f"(RAG retrieval error: {e})"


# =========================
# Prompt Builder
# =========================
def build_prompt(user_message: str, extra_context: str = "", history=None) -> str:
    history = history or []

    if extra_context.strip():
        system = (
            "You are TravelBuddy — a friendly, knowledgeable, and reliable travel assistant. "
            "Use any supplied knowledge base, file, or image content plus the user's message to answer. "
            "Extract dates, locations, booking codes, prices, or travel details. "
            "If conflicts exist, politely ask for confirmation. "
            "Redact sensitive info (credit cards, passport numbers). "
            "Give concise and actionable answers (bullet points, steps, itineraries)."
        )
        parts = [system, "\n---\nConversation:\n"]
        for u, a in history:
            parts.append(f"User: {u}\nAssistant: {a}\n")
        parts.append(f"User: {user_message}\n📎 Context:\n{extra_context}\nAssistant:")
        return "\n".join(parts)
    else:
        system = (
            "You are TravelBuddy — a friendly and conversational assistant. "
            "You can chat casually or answer general questions. "
            "When files/images/knowledge are provided, use them for travel help. "
        )
        parts = [system, "\n---\nConversation:\n"]
        for u, a in history:
            parts.append(f"User: {u}\nAssistant: {a}\n")
        parts.append(f"User: {user_message}\nAssistant:")
        return "\n".join(parts)

# =========================
# File / OCR Helpers
# =========================
def extract_text_from_file(file_path: str) -> str:
    text = ""
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
        elif ext == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            text = df.to_string()
        else:
            text = f"Unsupported file format: {ext}"
    except Exception as e:
        text = f"(Error reading file: {e})"
    return text.strip()


def ocr_text_from_image(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"(OCR error: {e})"


# =========================
# Qwen Processing
# =========================
def process_with_qwen(prompt: str, image_path: str = None) -> str:
    """Process text or image+text with Qwen VL model."""
    try:
        if image_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": Image.open(image_path).convert("RGB")},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            
        prompt_text = qwen_processor.apply_chat_template(messages, add_generation_prompt=True)
        
        if image_path:
            inputs = qwen_processor(
                text=[prompt_text], 
                images=[Image.open(image_path).convert("RGB")], 
                return_tensors="pt"
            ).to(qwen_model.device)
        else:
            inputs = qwen_processor(
                text=[prompt_text], 
                return_tensors="pt"
            ).to(qwen_model.device)
            
        with torch.inference_mode():
            generated_ids = qwen_model.generate(
                **inputs, 
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        output = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output.strip()
    except Exception as e:
        return f"(Qwen error: {e})"

# =========================
# Vision-Language Analyzer
# =========================
def qwen_vl_answer(image_path: str, user_message: str) -> str:
    global img_processor, img_model
    try:
        travel_prompt = (
            "You are TravelBuddy — a reliable travel assistant.\n"
            "Analyze the image for travel details: dates, locations, tickets, booking info, prices, landmarks. "
            "If text is present, transcribe it. Return a concise structured analysis.\n\n"
            f"User question: {user_message}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": travel_prompt},
                    {"type": "image", "image": image_path},
                ],
            }
        ]
        prompt_text = img_processor.apply_chat_template(messages, add_generation_prompt=True)
        pil_image = Image.open(image_path).convert("RGB")
        inputs = img_processor(text=[prompt_text], images=[pil_image], return_tensors="pt").to(img_model.device)
        with torch.inference_mode():
            generated_ids = img_model.generate(**inputs, max_new_tokens=400)
        output = img_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output.strip()
    except Exception as e:
        return f"(VLM error: {e})"

# =========================
# Chat Function
# =========================
def chat_fn(message, history, file=None, image=None):
    if not message or message.strip() == "":
        return "Please enter a question."

    message = message.strip()
    extra_context = ""

    # 1. Knowledge base retrieval
    rag_context = retrieve_context(message)
    if rag_context:
        extra_context += f"\n[Knowledge Base]:\n{rag_context}"

    # 2. File upload
    if file is not None:
        try:
            extra_context += "\n\n" + extract_text_from_file(file.name)
        except Exception as e:
            extra_context += f"\n(Note: Could not extract text from file: {e})"

    # 3. Image
    if image is not None:
        try:
            image_analysis = qwen_vl_answer(image, message)
            extra_context += f"\n[Image Analysis]:\n{image_analysis}"
        except Exception as e:
            extra_context += f"\n(Note: Could not analyze image: {e})"

    # History
    conversation_history = []
    if history:
        for pair in history:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                conversation_history.append((pair[0], pair[1]))

    # Prompt
    prompt = build_prompt(message, extra_context=extra_context, history=conversation_history)

    # Generate response using Qwen
    try:
        text = process_with_qwen(prompt)
        if not text:
            text = "Sorry, I couldn't find a good answer. Can you clarify?"
        elif "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()
    except Exception as e:
        text = f"(Error during generation: {e})"

    return text

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 TravelBuddy — AI Travel Assistant with RAG")
    gr.Markdown("Ask travel questions, upload files/images, and query your knowledge base.")

    chatbot = gr.Chatbot()
    user_message = gr.Textbox(placeholder="Ask me about your trip...", lines=2)

    file_input = gr.File(label="Upload a file (PDF, DOCX, CSV, TXT)", type="filepath")
    image_input = gr.Image(type="filepath", label="Upload an image")

    submit_btn = gr.Button("Ask TravelBuddy")
    state = gr.State([])

    def process(message, history, file, image):
        reply = chat_fn(message, history, file, image)
        history.append((message, reply))
        return history, history, None, None

    submit_btn.click(
        process,
        [user_message, state, file_input, image_input],
        [chatbot, state, file_input, image_input],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)