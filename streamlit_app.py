

import streamlit as st
from llama_cpp import Llama
import os

# Load model (load once and cache)
@st.cache_resource
def load_model():
    model_path = os.path.expanduser("~/mistral-model/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    return Llama(model_path=model_path, n_ctx=1024, n_gpu_layers=20)

llm = load_model()

st.title("💬 Mistral Chat — Local & Offline")

# User input
user_input = st.text_area("Enter your instruction", height=150)

if st.button("Run"):
    if user_input.strip():
        prompt = f"### Instruction:\n{user_input}\n\n### Response:"
        with st.spinner("Generating response..."):
            output = llm(prompt, max_tokens=200)
            st.markdown("**Response:**")
            st.write(output["choices"][0]["text"].strip())
    else:
        st.warning("Please enter some text.")
