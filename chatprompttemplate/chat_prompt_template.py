from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from llm_service import  get_llm
import streamlit as st

# Get Ollama instance
llama = get_llm()
#Prompt using Chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [('system', "You are an professional profile reviewer, who can analyser profiles based on the resume details and match them with the respective fields "),
     MessagesPlaceholder(variable_name="chat_history"),
     ('human', "{input}")]
)

chain = prompt | llama

history = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# question = input("Enter a message:")
# response = chain_with_history.invoke({"input": question}, {"configurable": {"session_id": "abc123"}})
# print(response)
