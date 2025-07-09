from tempfile import template

from langchain_core.prompts import PromptTemplate

from llm_service import get_llm
import streamlit as st


input_variables = ['company', 'position', 'strengths', 'weaknesses']
template = """You are a career coach. Provide tailored interview tips for the position of {position}
at {company}.
Highlight your strengths in {strengths} and weaknesses in {weaknesses} and prepare questions accordingly."""
prompt_template = PromptTemplate(input_variables=input_variables,
                                 template= template)
llama = get_llm()
st.title("Interview Helper")
company = st.text_input("Enter the company:")
position = st.text_input("Enter the position:")
strengths = st.text_input("Enter the strengths:")
weaknesses = st.text_input("Enter the weaknesses:")

input_dict = {
    'company': company,
    'position': position,
    'strengths': strengths,
    'weaknesses': weaknesses,
}
chain = prompt_template | llama

if company and position and strengths and weaknesses:
    response = chain.invoke(input_dict)
    st.write(response)


