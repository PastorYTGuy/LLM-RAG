import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import tools
from database import VectorDB
from pathlib import Path
import logging
import re
from datetime import datetime
import json

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RAG_ENGINE")

LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

def log_event_to_jsonl(data: dict):
    """Zapisuje zdarzenie do pliku .jsonl w formacie jednej linii na wpis."""
    try:
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)

        filename = f"logs_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        log_file = os.path.join(LOGS_DIR, filename)

        payload = {
            "timestamp": datetime.now().isoformat(),
            **data
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()

        logger.info(f"Log zapisany pomyślnie w {log_file}")

    except Exception as e:
        logger.error(f"BŁĄD ZAPISU LOGA: {e}")



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
    from google.genai import errors

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    print("Using Gemini LLM:", GEMINI_MODEL)






def retrieve(question: str, k=10):
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


def ask_pipeline(question: str, k=10, use_functions=True):
    logger.info(f"Nowe zapytanie: {question}")
    is_safe, reason = tools.input_guardrail(question)
    if not is_safe:
        logger.warning(f"Guardrail zablokował pytanie: {reason}")
        return {"status": "BLOCKED", "final": reason, "meta": {}}






    result_to_return = {}

    if not USE_LOCAL:
        try:

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
            tool_config = None
            if any(op in question for op in ["+", "-", "*", "/", "ile to", "oblicz"]):
                tool_config = {"function_calling_config": {"mode": "ANY"}}

            try:
                hits = retrieve(question, k=k)
                ctx, meta = pack_context(hits)
                logger.info(f"Pobrano {len(hits)} dokumentów z bazy.")
            except Exception as e:
                logger.error(f"Błąd bazy wektorowej: {e}")
                return {"status": "error", "final": "Problem z dostępem do wiedzy.", "meta": {}}

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"Context:\n{ctx}\n\nQuestion: {question}",
                config={
                    "system_instruction": system_instruction,
                    "tools": tools.FUNCTIONS if use_functions else None,
                    "tool_config": tool_config
                }
            )

            candidate = response.candidates[0]

            function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

            if function_calls:
                fc = function_calls[0]

                tool_result = tools.validate_and_dispatch(fc.name, fc.args)
                result_to_return = {"status": "ok", "final": str(tool_result.get("output") or tool_result.get("reason")), "meta": meta}
            else:
                final_text = response.text
                result_to_return = {"status": "ok", "final": final_text, "meta": meta}
        except errors.ClientError as e:
            status_code = getattr(e, 'status_code', None)
            error_message = str(e).upper()
            if status_code == 429 or "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                logger.error("!!! LIMIT API WYCZERPANY (429) !!!")
                result_to_return = {
                    "status": "QUOTA_EXCEEDED",
                    "final": "Limit zapytań Gemini wyczerpany. Spróbuj ponownie za minute albo przełącz się na model lokalny!",
                    "meta": meta
                }
            else:
                logger.error(f"Inny błąd Gemini API: {e}")
                result_to_return = {"status": "error", "final": "Coś poszło nie tak z połączeniem do bazy wiedzy...", "meta": meta}


    else:

        system_instruction = (
            "You are a secure assistant with the persona of a friendly gaming geek who loves sharing knowledge\n"
            "Strict Compliance Rules:\n"
            "Counting by yourself is FORBIDDEN, if prompt has math problems, ONLY use CALL:calculator\n"
            "Example: User: Ile to 10 + 20?\n"
            "Assistant: CALL: calculator(a=10, b=20)\n"
        )

        try:
            hits = retrieve(question, k=k)
            ctx, meta = pack_context(hits)
            logger.info(f"Pobrano {len(hits)} dokumentów z bazy.")
        except Exception as e:
            logger.error(f"Błąd bazy wektorowej: {e}")
            return {"status": "error", "final": "Problem z dostępem do wiedzy.", "meta": {}}

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        match = re.search(r"(?:^|\n)\s*CALL:\s*(\w+)\((.*)\)", response_text)

        if match:
            t_name = match.group(1)
            t_args_raw = match.group(2)

            t_args = {}
            pairs = re.findall(r"(\w+)\s*=\s*['\"]?([-+]?\d*\.?\d+|[^'\",\s)]+)['\"]?", t_args_raw)
            for key, val in pairs:
                try:
                    if re.match(r"^-?\d+(\.\d+)?$", val):
                        t_args[key] = float(val) if '.' in val else int(val)
                    else:
                        t_args[key] = val
                except ValueError:
                    t_args[key] = val
            logger.info(f"Wykryto żądanie FC: {t_name}({t_args})")
            tool_result = tools.validate_and_dispatch(t_name, t_args)

            tool_output = tool_result.get('output', {}).get('result') or tool_result.get('output') or tool_result.get('reason')
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Otrzymano wynik z narzędzia {t_name}: {tool_output}. Skomentuj to teraz krótko jako gaming geek i podaj wynik."})

            final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            final_inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
            final_outputs = model.generate(**final_inputs, max_new_tokens=200, temperature=0.5, do_sample=True)

            final_text = tokenizer.decode(final_outputs[0][len(final_inputs.input_ids[0]):], skip_special_tokens=True)
            result_to_return = {"status": "ok", "final": final_text, "meta": meta}
        else:
            if any(char.isdigit() for char in question) and ("+" in question or "dodaj" in question):
                logger.warning("Model zignorował kalkulator! Próba wymuszenia.")
            result_to_return = {"status": "ok", "final": response_text, "meta": meta}

    log_event_to_jsonl({
        "question": question,
        "final_response": result_to_return.get("final"),
        "status": result_to_return.get("status"),
        "meta": result_to_return.get("meta")
    })

    return result_to_return