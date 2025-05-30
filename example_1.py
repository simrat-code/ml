from urllib import response
import ollama
import chromadb


documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas, camels and dummy",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

model_list = ["all-minilm", "codellama", "llama3.2"]
display_list = "\n".join(["\t"+str(i)+" - "+x for i, x in enumerate(model_list)])
choice = int(input(f"{display_list}\nselect the model: "))
model = model_list[choice]
print(f"[=] using model: {model}")

client = chromadb.Client()
collection = client.get_or_create_collection(name="docs")

#
# store each document in a vector embedding db
#
for i, d in enumerate(documents):
    # response = ollama.embeddings(model=model, prompt=d)   # embeddings() is old and may get obsolete in future
    response = ollama.embed(model=model, input=d)
    embeddings = response["embeddings"]
    print(type(embeddings))
    print(f"embedding {i}: [{len(embeddings)}][{len(embeddings[0])}]: {embeddings[0][:5]}")

    collection.add(
        ids = [str(i)],
        embeddings=embeddings,
        documents=[d]
    )

#
# user prompt
#
prompt = "What animals are llama related to?"

#
# generate embedding for prompt and retrieve most relevant doc
#
response = ollama.embed(model=model, input=prompt)
results = collection.query(
    query_embeddings = response["embeddings"],
    n_results=1
)

data = results['documents'][0][0]
print(f"data: {data}")

#
# generate answer from retrieved document
#
output = ollama.generate(
    model="llama3.2",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)
print(f"response: {output['response']}")
# end
