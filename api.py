from fastapi import FastAPI

from main import get_response
app = FastAPI(title="BGE+RAG search")
@app.get("/agent/{query}")
def get_ai_response(query:str):
    response = get_response(query)
    return {'response': response}