# Synthetic Gen Agent
## Opis
Znajdziesz tutaj kod, który pozwala przy użyciu agentów budować zdywersyfikowany zbiór danych. Wykorzystuje modele OpenAI oraz polski LLM Bielik do generowania danych syntetycznych.

### Role
W kodzie można wyszczególnić następujące role:
- Manager: główny model, który ma dostęp do narzędzi
- DataScientist: narzędzie, które odpowiada za analizę danych
- Pomysłodawca: narzędzie, które może wygenerować pomysły na teksty
- Pisarz: narzędzie, które może wygenerować teksty o podanych atrybutach

### Stuktura
```
.
├── models/ # definicje obiektów
│ └── models.py
├── prompts/ # prompty wykorzystywane w rozwiązaniu
│ ├── cel.yaml
│ ├── data_scientist.yaml
│ ├── generate_text.yaml
│ ├── ideas_creator.yaml
│ ├── manager.yaml
│ └── map_reduce.yaml
├── tools/ # Dostępne narzędzia
│ ├── AnalyzeMetrics.py
│ ├── GenerateText.py
│ └── IdeasCreator.py
├── utils/ # Dodatkowe funkcjonalności
│ ├── map_reduce.py
│ └── utils.py
├── .env # Zmienne środowiskowe - klucz OpenAI
├── .gitignore
├── LICENSE
├── main.py # Główny skrypt
├── pyproject.toml # Konfiguracja projektu
├── README.md
├── settings.py # Główny plik z ustawieniami
└── uv.lock # Lock file dla bibliotek
```

### Dodatki
Projekt wykorzystuje podejście MapReduce do generowania podsumowań wygenerowanych tekstów. ardzo się inspirowałem: https://arxiv.org/pdf/2410.09342 oraz https://github.com/thunlp/LLMxMapReduce?tab=readme-ov-file.

# Jak uruchomić projekt?
1. Dodaj swój klucz OpenAI do pliku zmiennych środowiskowych .env
2. Zainstaluj wymagane biblioteki
```
# Przy użyciu uv
uv sync

# Przy użyciu pip
pip install -r requirements.txt
```
3. Uruchom rozwiązanie
```
python main.py
```