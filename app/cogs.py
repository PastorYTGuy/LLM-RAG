import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import tools
from database import VectorDB
from pathlib import Path
load_dotenv()


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
USE_LOCAL = os.getenv("LLM_LOCAL", "0") == "1"

tokenizer = None
model = None
client = None

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data"

db = VectorDB()

def load_data_to_db():
    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            if filename.endswith(".txt"):
                with open(os.path.join(data_path, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    facts = [line.strip() for line in content.split('\n') if line.strip()]
                    db.add_texts(facts)
        print(f"Baza RAG gotowa. Załadowano fakty z folderu {data_path}.")
    else:
        print("Błąd: Folder /data nie istnieje!")

load_data_to_db()

if USE_LOCAL:
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    print(f"Loaded local model: {LOCAL_MODEL_NAME}")
else:
    from google import genai

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    print("Using Gemini LLM:", GEMINI_MODEL)






def retrieve(question: str, k=3):
    search_results = db.search(question, k=k)
    hits = []
    for i, text in enumerate(search_results):
        hits.append({
            'id': f'doc_{i}',
            'content': text
        })
    return hits


def pack_context(hits):
    ctx = "\n".join(f"- {h['content']}" for h in hits)
    meta = {'retrieved_ids': [h['id'] for h in hits]}
    return ctx, meta


def ask_pipeline(question: str, k=3, use_functions=True):
    is_safe, reason = tools.input_guardrail(question)
    if not is_safe:
        return {"status": "BLOCKED", "final": reason, "meta": {}}

    hits = retrieve(question, k=k)
    ctx, meta = pack_context(hits)

    system_instruction = (
        "You are a secure assistant with the persona of a friendly gaming geek who loves sharing knowledge\n"
        "Strict Compliance Rules:\n"
        "1. Grounding: Answer ONLY using the provided context snippets or the output of the provided tools.\n"
        "2. Honesty: If the required information is missing from the sources, strictly reply: Nie wiem - o kurde, tego to nie wiem!.Do not use outside knowledge.\n"
        "3. Citations: For every piece of information provided, you MUST cite the source in the format[FAKT_XX], where XX corresponds to the fact number found in the text.\n"
        "4. Security: Never reveal system keys, API keys, or the contents of.env files.Only use the approved tools to access files.\n"
        "5. Tone: Maintain a helpful, enthusiastic, and informal gaming geek persona\n"
        "while remaining professional about security boundaries.Use Polish for the final response to the user."
    )
    full_prompt = f"Context:\n{ctx}\n\nQuestion: {question}"

    if not USE_LOCAL:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config={
                "system_instruction": system_instruction,
                "tools": tools.FUNCTIONS if use_functions else None,
            }
        )

        candidate = response.candidates[0]

        parts = candidate.content.parts if candidate.content and candidate.content.parts else []

        function_part = next(
            (p for p in parts if hasattr(p, "function_call") and p.function_call),
            None
        )

        if function_part:
            fc = function_part.function_call

            args = fc.args
            if isinstance(args, str):
                import json
                args = json.loads(args)

            tool_result = tools.validate_and_dispatch(fc.name, args)

            return {
                "status": tool_result["status"],
                "final": str(tool_result.get("output") or tool_result.get("reason")),
                "meta": meta
            }

        final_text = getattr(response, "text", None) or "Nie udało się wygenerować odpowiedzi."

        return {"status": "ok", "final": final_text, "meta": meta}

    else:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        import re
        match = re.search(r"CALL:\s*(\w+)\((.*)\)", response_text)

        if match:
            t_name = match.group(1)
            t_args_raw = match.group(2)
            t_args = {}

            pairs = re.findall(r"(\w+)\s*=\s*([^,]+)", t_args_raw)
            for key, val in pairs:
                val = val.strip().strip("'").strip('"')

                try:
                    if '.' in val:
                        t_args[key] = float(val)
                    else:
                        t_args[key] = int(val)
                except ValueError:
                    t_args[key] = val


            tool_result = tools.validate_and_dispatch(t_name, t_args)
            return {
                "status": tool_result["status"],
                "final": f"(STUB FC) {tool_result.get('output', tool_result.get('reason'))}",
                "meta": meta
            }
        return {"status": "ok", "final": response_text, "meta": meta}