# 🧠 Local Mistral LLM on MacBook (Apple Silicon)

Run the powerful open-source [Mistral 7B Instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) language model **completely offline** on your MacBook Pro using Metal GPU acceleration.

No APIs. No token limits. No cost. Just raw, local intelligence 💻⚡

---

## 🚀 Features

- 🐍 Python interface using `llama-cpp-python`
- ⚡ Optimized for Apple Silicon (M1, M2, M3, M4)
- 🧠 Run quantized Mistral 7B (4-bit) models locally
- 📝 Easily modify prompts, automate tasks, and build cool stuff

---

## 📦 Requirements

- macOS with Apple Silicon (M1 or newer)
- Python 3.10+
- Metal GPU support (comes with macOS)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

---

## 🛠️ Setup Instructions

### 1. Clone the repo and create your project folder

```bash
mkdir -p ~/Documents/python-projects/mistral-demo
cd ~/Documents/python-projects/mistral-demo
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model

```bash
mkdir -p ~/mistral-model
cd ~/mistral-model

curl -L -o mistral-7b.Q4_K_M.gguf \
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

```

## (Optional)

## Generate Custom Regional Language Dataset Using Google Translate API 

### 1. Activate Google Translate API (First $300 USD Free)


### 2. Install google-cloud-translate
```bash
    pip install google-cloud-translate
```

### 3. Run the script
```bash
python3 google_translate.py
```

