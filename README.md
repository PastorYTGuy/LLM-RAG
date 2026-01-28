Gaming Geek RAG & Tool Assistant

System asystenta opartego na RAG z Function Calling i Guardrailami

### Wymagania
* Python 3.10+
* Klucz API Google AI Studio

### Setup
* stworzenie środowiska venv
  * np. python -m venv venv
* instalacja zależności
  * pip install -r requirements.txt

### uruchomienie
* w celu sprawdzenia poprawności działania, uruchom test ewaluacji i sprawdź plik tekstowy raport.txt w folderze '/app'
  * python evaluation.py
* Główna część programu jest uruchamiana poprzez main.py i wykorzystuje FastAPI
  * python main.py
    * pod adresem: http://localhost:8000/docs

### data flow
* każde z zapytań przechodzi przez testy heurystyki, jeśli system wykryje próbę manipulacji wyniku poprzez zakazane słowa lub inne metody, wynik jest blokowany
* VectorDB wykorzystując FAISS i model all-MiniLM-L6-v2 wyszukuje top k najlepszych wyników
* W zależności od konfiguracji flagi USE_LOCAL w .env prompt jest przetwarzany przez Gemini-2.5-flash lub Qwen. LLM sprawdza czy wykorzystać dostępne narzedzia, czy odpowiedzieć bezpośrednio
* Jeśli model zdecyduje się na wykorzystanie narzędzi, przechodzi to walidacje