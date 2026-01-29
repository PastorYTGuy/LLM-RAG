import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cogs import ask_pipeline, log_event_to_jsonl
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")


app = FastAPI(title="LLM RAG Guardian API")



class QuestionRequest(BaseModel):
    question: str
    use_functions: bool = True


@app.get("/")
def read_root():
    return {"status": "running", "message": "LLM API is ready. Go to /docs for Swagger."}


@app.post("/ask")
async def ask(request: QuestionRequest):
    start_time = time.time()
    try:
        result = ask_pipeline(request.question, use_functions=request.use_functions)
        duration = time.time() - start_time
        log_event_to_jsonl({
            "endpoint": "/ask",
            "question": request.question,
            "status": result.get("status"),
            "duration_sec": round(duration, 3),
            "guardrail_triggered": result.get("status") == "BLOCKED"
        })

        if result.get("status") == "BLOCKED":
            return {
                "answer": result["final"],
                "guardrail_triggered": True,
                "meta": result.get("meta", {})
            }
        return result

    except Exception as e:
        logger.error(f"Krytyczny błąd podczas obsługi pytania: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Wystąpił wewnętrzny błąd serwera.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
