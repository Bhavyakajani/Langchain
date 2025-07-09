from langchain_ollama import ChatOllama
import  os
from langchain import hub
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools


llama = ChatOllama(model='llama3.2')

prompt_agent = hub.pull("hwchase17/react")

tools = load_tools(["wikipedia", "ddg-search"])

agent = create_react_agent(llama, tools, prompt_agent)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.title("AI agent")

task  = st.text_input("Assign me a Task")

if task:
    response_agent = agent_executor.invoke({"input":task})
    st.write(response_agent["output"])