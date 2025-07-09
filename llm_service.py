from pprint import pprint

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain.globals import set_debug
import json

import document_processing as dp
import streamlit as st


# set_debug(True)
# llama = OllamaLLM(model='llama3.2', base_url="http://127.0.0.1:11434")

FILE_PATH = 'C:\\Users\\tonyj\\OneDrive\\Desktop\\Sem8\\Thesis\\RAG_project\\data\\Bhavya Kajani-1.pdf'
file = st.file_uploader("Upload PDF/JPG/pptx:")

def get_llm():
    return OllamaLLM(model='llama3.2', base_url="http://127.0.0.1:11434")

def get_embedding_llm():
    return OllamaEmbeddings(model='llama3.2')

def ask_llm(prompt):
    llama = get_llm()
    return llama.invoke(prompt)

fields = """- Skills: [list]
    - Experience: [years, roles]
    - Education: [degree, university]
    - Name: [name]
    - Contact: [contact_info]"""
format = """{
  "name": "",
  "contact_number": "",
  "email": "",
  "skills": [
    "Python",
    "Machine Learning",
    "Data Analysis",
    "SQL",
    "JavaScript"
  ],
  "educations": [
    {
      "institution": "",
      "start_date": "",
      "end_date": "",
      "location": "",
      "degree": ""
    }
  ],
  "work_experiences": [
    {
      "company": "",
      "start_date": "",
      "end_date": "",
      "location": "",
      "role": ""
    }
  ],
  "YoE": ""
}"""

context = dp.normalize_text(dp.load_file(FILE_PATH))

def get_prompt():
    return f"""
        You are an AI model with excellent skills in extracting information for a profile based on the fields, format and text below:
        Fields: {fields}
        Sample JSON format:
        Format: {format}
        Text: {context}
    - Do not make up any information, leave if the field empty if information is not present.    
    - Answer in JSON format. No further explanation nor natural language outside of JSON can be present. 
    - Do not include any other information.
    - Give the output without any escape characters and only in JSON format.    
    """


prompt_template = PromptTemplate(
    input_variables=['fields', 'format', 'context'],
    template="""
        You are an AI model with excellent skills in extracting information for a profile based on the fields, format and text below:
        Fields: {fields}
        Format: {format}
        Text: {context}
    - Do not make up any information, leave if the field empty if information is not present.    
    - Answer in JSON format. No further explanation nor natural language outside of JSON can be present. 
    - Do not include any other information.
    - Give the output without any escape characters and only in JSON format.

    Sample JSON format:
    
    """)



#
# print(context)
llama = get_llm()
input_variables_dict = {
    'fields': fields,
    'format': format,
    'context': context
}
chain = prompt_template | llama | JsonOutputParser()
# response = llama.invoke("Extract from this résumé in natural language:")
response = chain.invoke(input_variables_dict)
pprint(response)
pprint(response['name'])
# print(parser.parse(response))