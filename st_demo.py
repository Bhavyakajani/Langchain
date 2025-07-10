from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
# In case of OpenAI replace the above import with the one below:
from langchain_openai import ChatOpenAI
import streamlit as st

st.title("Chat with Ollama/OpenAI")

llama = ChatOllama(model='llama3.2')
#  chatGPT = ChatOpenAI (model='', api_key = your_api_key)

with st.sidebar:
    st.title("Provide your API key:")
    API_KEY = st.text_input("OpenAI api key", type = "password")
    model_choice = st.selectbox("Choose an LLM", ['llama3.2', 'OpenAI'])
    if model_choice == 'OpenAI':
        llama = ChatOpenAI(model='gpt-3.5-turbo', api_key = API_KEY)
    else:
        st.info("Using Ollama by default")

if not API_KEY:
    st.info("Using Ollama by default")


st.subheader("Chat Interface")

question = st.text_input("Enter a question:")

if question:
    try:
        response_ollama = llama.invoke(question) #replace llama -> chatGPT
        st.write(response_ollama)
    except Exception as e:
        st.error(f"Error: {e}")