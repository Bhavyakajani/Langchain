from llm_service import get_llm, response, get_embedding_llm



llama = get_embedding_llm()
text = input("Enter a message:")
list_text = [
    'Hi', 'How are you?',
    'Cats are funny.'
]
response_embed = llama.embed_documents(list_text)
for item in response_embed:
    print(f"{item}\n")
print(len(response_embed))

