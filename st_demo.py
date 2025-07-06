from langchain_ollama import ChatOllama
# In case of OpenAI replace the above import with the one below:
# from langchain_openai import ChatOpenAI
import streamlit as st

st.title("Chat with Ollama: llama3.2:3b")

llama = ChatOllama(model='llama3.2')
#  chatGPT = ChatOpenAI (model='', api_key = your_api_key)

with st.sidebar:
    st.title("Provide your API key:")
    API_KEY = st.text_input("OpenAI api key", type = "password")
if not API_KEY:
    st.info("Using Ollama by default")


st.subheader("(Not a vision model)")

question = st.text_input("Enter a question:")

if question:
    response_ollama = llama.invoke(question) #replace llama -> chatGPT
    st.write(response_ollama)