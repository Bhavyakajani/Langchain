from langchain_ollama import ChatOllama
import  os
from langchain import hub
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import base64


def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

llama = ChatOllama(model='llama3.2-vision:11B')
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can identify a Landmark"),
        ("human",
         [
             {"type": "text", "text": "return landmark name"},
             {
                 "type": "image_url",
                 "image_url": {
                     "url": f"data:image/jpeg;base64,""{image}",
                     "detail": "low",
                 },
             },
         ],
        ),
    ]
)

chain = prompt_template | llama

st.title("Landmark Helper")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Question related to the Landmark")

task = None

if question:
    image = encode_image(uploaded_file)
    resource = chain.invoke({"image": image})
    task = question+resource.content

prompt_landmark = hub.pull("hwchase17/react")

tools = load_tools(["wikipedia", "ddg-search"])

agent = create_react_agent(llama, tools, prompt_landmark)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if task:
    response_agent = agent_executor.invoke({"input":task+" without explanation"})
    st.write(response_agent["output"])