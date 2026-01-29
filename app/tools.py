import logging

logger = logging.getLogger(__name__)

def input_guardrail(question: str) -> (bool, str):

    forbidden_patterns = [
        "zapomnij", "forget", "ignore previous", "instrukcja",
        "system prompt", "dan mode", "do anything", "zrób coś innego",
        "jesteś teraz", "be a dev", "reveal", "secret keys", "debug", "secret key"
    ]

    q_lower = question.lower()



    #Prompt Injection
    for pattern in forbidden_patterns:
        if pattern in q_lower:
            return False, f"Zablokowano: Wykryto próbę manipulacji (fraza: '{pattern}')."

    #próby dostępu do plików systemowych
    if any(secret in q_lower for secret in [".env", "config.py", "secrets", "/etc/", "/"]):
        return False, "Zablokowano: Próba dostępu do plików systemowych."

    return True, ""


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = (BASE_DIR / "data").resolve()


def files_search_impl(pattern: str):
    """Searches for file names in the allowed data directory matching a pattern."""
    if ".." in pattern or "/" in pattern or "\\" in pattern:
        return {"status": "blocked", "reason": "unsafe_characters_in_pattern"}

    if ".env" in pattern.lower():
        return {"status": "blocked", "reason": "access_denied_to_secrets"}


    return {'files': [f'found_{pattern}_1.txt']}


def read_document_impl(filename: str):
    """Reads the content of a specific file from the data directory. Use this after finding a file name."""
    try:

        target_path = (BASE_DATA_DIR / filename).resolve()

        if not target_path.is_relative_to(BASE_DATA_DIR):
            return {"status": "blocked", "reason": "path_traversal_attempted"}

        if target_path.name == ".env" or target_path.suffix in [".py", ".sh"]:
            return {"status": "blocked", "reason": "file_type_restricted"}

        if not target_path.exists():
            return {"error": "File not found"}

        with open(target_path, "r", encoding="utf-8") as f:
            return {"content": f.read(1000)}

    except Exception as e:
        return {"error": str(e)}



def kb_lookup_impl(query: str, top_k: int = 3):
    """Searches the knowledge base for a specific query and returns top_k results."""
    return {'hits': [f'KB item about {query} #{i}' for i in range(1, top_k + 1)]}


def calculator_impl(a: float, b: float):
    """Performs a mathematical addition of two numbers: a and b."""
    return {'result': a + b}


def validate_and_dispatch(tool_name: str, args: dict):
    print(f"DEBUG: Wywołanie narzędzia '{tool_name}' z argumentami: {args}")

    if tool_name not in TOOL_REGISTRY:
        return {"status": "security_blocked", "reason": f"Tool '{tool_name}' is not in allowlist."}


    expected_schema = TOOL_SCHEMAS.get(tool_name, {})
    validated_args = {}

    try:
        for param, expected_type in expected_schema.items():
            if param not in args:
                return {"status": "validation_error", "reason": f"Missing argument: {param}"}


            validated_args[param] = expected_type(args[param])

        result = TOOL_REGISTRY[tool_name](**validated_args)
        return {"status": "success", "output": result}
    except (ValueError, TypeError) as e:
        logger.warning(f"Błąd walidacji parametrów dla {tool_name}: {e}")
        return {"status": "error", "reason": f"Błędny typ danych dla {tool_name}"}
    except Exception as e:
        logger.error(f"Krytyczny błąd narzędzia {tool_name}: {e}")
        return {"status": "error", "reason": "Błąd wewnętrzny narzędzia."}


FUNCTIONS = [kb_lookup_impl, files_search_impl, calculator_impl, read_document_impl]

TOOL_REGISTRY = {
    "kb_lookup": kb_lookup_impl,
    "files_search": files_search_impl,
    "calculator": calculator_impl,
    "read_document": read_document_impl
}

TOOL_SCHEMAS = {
    "calculator": {"a": float, "b": float},
    "read_document": {"filename": str},
    "files_search": {"pattern": str},
    "kb_lookup": {"query": str, "top_k": int}
}