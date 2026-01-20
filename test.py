import os
from openai import OpenAI

print("API Key:", os.getenv("OPENAI_API_KEY"))

client = OpenAI()

respons = client.embeddings.create(
    input=["Hello world!"],
    model="text-embedding-3-small"
)
print("bfthfynjgy")
print(respons)
