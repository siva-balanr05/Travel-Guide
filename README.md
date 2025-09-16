# 🌍 TravelBuddy — AI-Powered Travel Assistant

TravelBuddy is an AI-powered travel assistant that helps users plan trips, analyze travel documents, extract booking details, and answer questions using **RAG (Retrieval-Augmented Generation)** + **Vision-Language Models** — all running locally on CPU.

---

## ✨ Features


- 📷 **AI Text Generation** **Image Analysis** — Extracts travel details (tickets, dates, prices) using [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct).
- 📚 **RAG Knowledge Base** — Loads custom `knowledge_base.pdf` for document-based Q&A.
- 📂 **File Support** — Reads PDF, DOCX, CSV, TXT files and uses them in context.
- 🖼 **OCR Support** — Extracts text from travel images with Tesseract OCR.
- 🌐 **Gradio Web UI** — Simple, interactive chat interface accessible in your browser.

---

## 🛠 Tech Stack

- **Python 3.10+**
- [PyTorch (CPU)](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Gradio](https://gradio.app/)

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/siva-balanr05/Travel-Guide.git
cd Travel-Guide/travelbuddy_cache
