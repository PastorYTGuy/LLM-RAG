from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cogs import ask_pipeline
import uvicorn

app = FastAPI(title="LLM RAG Guardian API")



class QuestionRequest(BaseModel):
    question: str
    use_functions: bool = True


@app.get("/")
def read_root():
    return {"status": "running", "message": "LLM API is ready. Go to /docs for Swagger."}


@app.post("/ask")
async def ask(request: QuestionRequest):
    try:

        result = ask_pipeline(request.question, use_functions=request.use_functions)

        if result.get("status") == "BLOCKED":

            return {

                "answer": result["final"],

                "guardrail_triggered": True

            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail="Wystąpił wewnętrzny błąd serwera.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
