from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from llm_service import get_llm, get_embedding_llm
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

llama = ChatOllama(model='llama3.2')
llama_embeddings = OllamaEmbeddings(model='llama3.2')


document = PyPDFLoader("Bhavya Kajani-1.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = text_splitter.split_documents(document)


vector_store = Chroma.from_documents(chunks, llama_embeddings)
retriever = vector_store.as_retriever()
prompt_rag = ChatPromptTemplate.from_messages(
    [('system', "You are an professional profile reviewer, who can analyser profiles based on the resume details and match them with the respective fields {context} "),
    MessagesPlaceholder(variable_name="chat_history"),
     ('human', "Context:\n{context}\n\n Question: {input}")]
)

history_aware_retriever = create_history_aware_retriever(llama, retriever, prompt_rag)
qa_chain = create_stuff_documents_chain(llama, prompt_rag)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

print("Chat with document")
question = input("Enter the question: ")

if question:
    response_rag = chain_with_history.invoke({"input": question}, {"configurable":{"session_id": "abc123"}})
    print(response_rag["answer"])

