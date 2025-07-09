from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from llm_service import get_llm, get_embedding_llm

llama = get_embedding_llm()

document = TextLoader("intro.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = text_splitter.split_documents(document)

db = Chroma.from_documents(chunks, llama)
text = """Purpose of file"""

embedding_vector = llama.embed_query(text)

docs = db.similarity_search_by_vector(embedding_vector)

print(docs)

