from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from llm_service import get_llm, get_embedding_llm
import base64, os

def encode_image(image_path):
    with open(image_path, mode="rb") as image_file:
        return base64.b64encode(image_file.read())

llama = get_llm()
image = encode_image("3049251.jpg")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can describe images."),
        ("human",
         [
             {"type": "text", "text": "{input}"},
             {
                 "image_url": {
                     "url": f"data:image/jpeg;base64,{image}",
                     "detail": "low",
                 },
             },
         ],
        ),
    ]
)

chain = prompt_template | llama

resource = chain.invoke({"input": "Explain"})

print(resource)