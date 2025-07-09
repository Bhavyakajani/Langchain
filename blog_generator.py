from tempfile import template

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llm_service import get_llm
import streamlit as st


input_variables_topic = ['topic']
topic_template = """You are a professional blogger.
            Create an outline for a blog post on the following topic: {topic}
            The outline should include: 
            1. Introduction
            2. Three main points with sub points
            3. Conclusion"""

input_variable_outline = ['outline']

introduction_template = """You are a professional blogger.
Write an engaging introduction based on the following outline: {outline}
The brief introduction should hook the reader and provide a brief about the blog"""
topic_prompt = PromptTemplate(input_variables=input_variables_topic,
                              template= topic_template)
introduction_prompt = PromptTemplate(input_variables=input_variable_outline,template= introduction_template)
llama = get_llm()
first_chain = topic_prompt | llama | StrOutputParser()
second_chain = introduction_prompt | llama | StrOutputParser()
final = first_chain |second_chain

st.title("Blog Generator")
topic = st.text_input("Topic")
input_dict = {
    'topic': topic,
}
if topic:
    # First generate the outline
    outline = first_chain.invoke({"topic": topic})

    # Then generate the introduction using the outline
    introduction = second_chain.invoke({"outline": outline})

    # Display both
    st.subheader("Outline")
    st.write(outline)

    st.subheader("Introduction")
    st.write(introduction)


