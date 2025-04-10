# pip install streamlit
# python -m streamlit run ./app.py
# pip install langchain
# pip install -qU langchain-ollama

import streamlit as st
from langchain_ollama import ChatOllama

st.title("**EventMaster Chatbot.**")

# main form
with st.form("llm-form"):
    text = st.text_area("Enter text")
    submit = st.form_submit_button("Submit")

# input and model selection
def generate_response(input_text):
    model = ChatOllama(model= "llama3.2:1b")
    response = model.invoke(input_text)
    return response.content

# chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# generate answer
if submit and text:
    with st.spinner("Generating response..."):
        response = generate_response(text)
        st.session_state['chat_history'].append({"user": text, "ollama": response})
        st.write(response)

# display history
for chat in st.session_state['chat_history']:
    st.write(f"**User:** {chat['user']}")
    st.write(f"**Ollama:** {chat['ollama']}")
    st.write("---")