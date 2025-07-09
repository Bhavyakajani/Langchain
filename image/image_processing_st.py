from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import base64


def encode_image(image_file):
        return base64.b64encode(image_file.getvalue()).decode('utf-8')

llama = ChatOllama(model='llama3.2-vision:11B')
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can describe images."),
        ("human",
         [
             {"type": "text", "text": "{input}"},
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
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question_image = st.text_input("Question:")
if question_image:
    image = encode_image(uploaded_file)
    resource = chain.invoke({"input": question_image, "image":image})
    st.write(resource)
