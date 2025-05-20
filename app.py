import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
import time
from generate import stream_generate
from prompt_templates import qna_prompt_template as prompt_template


model_path = "../models/phi_domain_bound_qna_finetuned_attempt_10/final_merged"


# ---- Load model and tokenizer (only once) ----
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    # quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False
    )
    model.eval()
    return tokenizer, model   
    

def main():
    # ---- Streamlit UI ----
    st.title("Medical Q & A")
    
    # Sidebar generation settings
    st.sidebar.header("Generation Settings")
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    top_k = st.sidebar.slider("Top-K", 0, 100, 50, 5)
    top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 1.0, 0.05)
    max_new_tokens = st.sidebar.slider("Max New Tokens", 10, 500, 150, 10)
    do_sample = st.sidebar.checkbox("Enable sampling", value=True)
    
    
    # Load model/tokenizer
    tokenizer, model = load_model()
    
    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle user prompt
    if prompt := st.chat_input("Ask a question..."):
        # Save user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Create response box
        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""
    
            # Stream model response
            for token in stream_generate(
                model,
                tokenizer,
                prompt_template.format(question=prompt),
                do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, 
                max_new_tokens=max_new_tokens
            ):
                full_response += token
                response_box.markdown(full_response + "▌")
    
            response_box.markdown(full_response)
    
        # Save bot response
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()