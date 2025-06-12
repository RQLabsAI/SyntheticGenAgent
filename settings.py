# Główne ustawienia projektu

# URL pod którym jest dostępna Ollama
OLLAMA_URL="http://localhost:11434"

# Wygenerowane pliki to JSONy zawierające dodatkowe dane, gdzie je zapisać?
JSON_OUTPUT_PATH="./json_generated_data"
# Można zapisywać również (dla łatwiejszego przeglądania wyników) tekst w postaci markdown
SAVE_MARKDOWN_FILES=True
# Jeżeli powyżej jest True, to gdzie zapisać pliki?
MARKDOWN_OUTPUT_PATH="./markdown"

# Maksymalna ilość kroków dla Managera
MAX_STEPS=100
# Co ile kroków tworzyć nowy plan?
PLANNING_INTERVAL=10

# Jaki model managera?
MANAGER_MODEL="gpt-4.1-mini"
# Jaki model ma być używany do analizowania zbioru danych
ANALYST_MODEL="o3"
# Model, który będzie wymyślać pomysły
IDEAS_CREATOR_MODEL="Bielik-11B-v2.6-Instruct:Q8_0"
# Model do generowania tekstów
TEXT_GENERATOR_MODEL="Bielik-11B-v2.6-Instruct:Q8_0"

# Pisarz jako zwrotke dla Managera podaje podsumowanie stworzonych tekstów, jaki model ma generować podsumowania?
MAP_REDUCE_MODEL = "Bielik-11B-v2.6-Instruct:Q8_0"
# Ile maksymalnie słów na jeden chunk można wysłać w MapReduce?
MAP_REDUCE_CONTEXT_WINDOW=10_000
