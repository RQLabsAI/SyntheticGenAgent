import hashlib
import importlib
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import requests
import spacy
import yaml
from ollama import Client
from pydantic import BaseModel
from smolagents import Tool

from models.models import Token
from settings import JSON_OUTPUT_PATH, SAVE_MARKDOWN_FILES, MARKDOWN_OUTPUT_PATH, OLLAMA_URL, TEXT_GENERATOR_MODEL, \
    MAP_REDUCE_MODEL, MAP_REDUCE_CONTEXT_WINDOW
from utils.map_reduce import MapReduce
from utils.utils import get_dataset_metrics, format_report

class Podsumowanie(BaseModel):
    podsumowanie: str
    ocena_jakosci: str

class GenerateNewDocument(Tool):
    name = "GenerateNewDocument"
    description = ('Użyj tego narzędzia do generowania tekstów w języku polskim na bazie ściśle określonych '
                   'parametrów. Przed użyciem tego narzędzia musisz dokładnie wiedzieć'
                   'jaką treść chcesz wygenerować. Na raz możesz wygenerować wiele tekstów. Generowanie na raz wielu '
                   'dokumentów (może to być nawet sto) jest optymalnym procesem. Jako odpowiedź otrzymasz krótkie '
                   'podsumowanie o stworzonych tekstach. Dowiesz się z niego o czym między innymi są, jakie mają '
                   'cechy charakterystyczne oraz dla jakiej grupy docelowej zostały stworzone. Każdy z tekstów będzie dodany do zbioru danych. '
                   'Każdy jeden tekst musi być bardzo konkretnie opisany. Warto na raz generować wiele zdywersyfikowanych tekstów. Każdy z nich jest '
                   'konkretnie zdefiniowany na dany temat, jednak razem tworzą coś zdywersyfikowanego. Funkcja ta generuje treść i nie ma dostępu do '
                   'obecnego zbioru danych. Jeden obiekt w liście argumentów = jeden tekst, który zostanie dodany do zbioru danych. Wszystkie atrybuty '
                   'obiektu muszą opisywać tylko jeden tekst. Korzystaj z listy wielu obiektów, by tworzyć zdywersyfikowane treści.')

    inputs = {
        "texts_to_generate": {
            "type": "array",
            "description": "Lista obiektów, które opisują pojedyńczy tekst do wygenerowania. Dla jednego obiektu zostanie stworzony jeden tekst. Każdy obiekt w liście musi być konkretnie i jednoznacznie zdefiniowany. Musi on być na jeden konkretny temat i pokrywać spójne punkty kluczowe. Najbardziej optymalnym podejściem jest generowanie jak najwięcej tekstów na raz. Jedno wywołanie funkcji może generować wiele tekstów z tej samej domeny.",
            "items": {
                "domain": {
                    "type": "string",
                    "description": 'Określ ściśle jeden, konkretny temat (domenę) tekstu. Przykłady: „medycyna”, „technologia”, „finanse”. Domena musi się składać z 1-4 słów. Nie używaj ogólników ani fraz typu „wiele domen” lub „różnorodne domeny”. Nie podawaj liczby tekstów do wygenerowania. Zdefiniuj jedną grupę docelową.',
                },
                "key_points": {
                    "type": "string",
                    "description": 'Podaj listę punktów kluczowych, które muszą zostać uwzględnione w tym tekście. Każdy kluczowy punkt opisz szczegółowo, aby stworzyć spójny dokument odpowiadający podanej dziedzinie. Podaj same konkrety. Nie używaj tutaj stwierdzeń o różnych czy też zdywersyfikowanych stylach. Musi to być jedna konkretna definicja tekstu.',
                },
                "additional_notes": {
                    "type": "string",
                    "description": 'Dodatkowe uwagi dotyczące stylu, słownictwa lub elementów, których należy unikać w tekście. Upewnij się, że komunikat jest precyzyjny, jednoznaczny i opisuje kwestie związane z jednym tekstem. Możesz też wymienić, jakich słów unikać.'
                },
                "target_audience": {
                    "type": "string",
                    "description": "Grupa docelowa. Zdefiniuj grupę docelową tekstu."
                },
                "style": {
                    "type": "string",
                    "description": "Zdefiniuj krótko w jakim stylu ma być tekst. Przykłady: 'formalny', 'techniczny i zaawansowany', 'luźny', 'wulgarny'."
                }
            }
        }
    }

    output_type = "array"

    def forward(self, texts_to_generate: List[dict]):
        # checks
        for i, text in enumerate(texts_to_generate):
            if "domain" not in text.keys():
                raise Exception(f"W obiekcie {i} brakuje klucza 'domain'")

            if "multi" in text["domain"] or "wiele" in text["domain"]:
                raise Exception(f"W obiekcie {i} brakuje specyficznej i konkretnej domeny.")

            if "additional_notes" not in text.keys():
                raise Exception(f"W obiekcie {i} brakuje klucza 'additional_notes'")

            if "key_points" not in text.keys():
                raise Exception(f"W obiekcie {i} brakuje klucza 'key_points'")

        texts_to_generate_text_log = []
        for text in texts_to_generate:
            texts_to_generate_text_log.append(f"Domena: {text['domain']}")
            texts_to_generate_text_log.append(f"Kluczowe punkty: {text['key_points']}")
            texts_to_generate_text_log.append(f"Uwagi: {text['additional_notes']}")
            texts_to_generate_text_log.append(f"Grupa docelowa: {text['target_audience']}")
            texts_to_generate_text_log.append(f"Styl: {text['style']}")
            texts_to_generate_text_log.append("")


        nlp = spacy.load('pl_core_news_sm')

        client = Client(
            host=OLLAMA_URL,
        )

        random.shuffle(texts_to_generate)

        in_tokens = 0
        out_tokens = 0

        prompts_generate_text = yaml.safe_load(importlib.resources.files("prompts").joinpath("generate_text.yaml").read_text())

        generated_texts = []
        for i, text in enumerate(texts_to_generate):
            response = client.chat(
                model=TEXT_GENERATOR_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": prompts_generate_text["system"].format(domain=text["domain"])
                    },
                    {
                        "role": "user",
                        "content": prompts_generate_text["user"].format(domain=text["domain"], key_points=text["key_points"], additional_notes=text["additional_notes"], target_audience=text["target_audience"], style=text["style"]),
                    }
                ]
            )
            in_tokens += response.prompt_eval_count
            out_tokens += response.eval_count

            output_text = response["message"]["content"]
            generated_texts.append(output_text)

            doc = nlp(output_text)

            # count words
            words = [token.text for token in doc if not token.is_punct and not token.is_space]
            number_of_words = len(words)

            # count sentences
            number_of_sentences = len(list(doc.sents))

            # average sentence length
            if number_of_sentences > 0:
                avg_sentence_length = number_of_words / number_of_sentences
            else:
                avg_sentence_length = 0

            avg_word_length = float(np.mean([len(word) for word in words]))
            pisarek_index = 100 - (avg_sentence_length + avg_word_length) * 5

            md5_hash = hashlib.md5(output_text.encode('utf-8')).hexdigest()

            doc = {
                "text": output_text,
                "domain": text['domain'],
                "tokens": [Token(token=str(t), pos=str(t.pos_)).json() for t in doc],
                "sentences": [[Token(token=str(t), pos=str(t.pos_)).json() for t in sentence] for sentence in doc.sents],
                "pisarek_index": pisarek_index,
                "style": text['style'],
                "target_audience": text['target_audience'],
                "model": TEXT_GENERATOR_MODEL
            }

            json_path = Path(JSON_OUTPUT_PATH)
            json_path.mkdir(parents=True, exist_ok=True)
            json_file_path = json_path / f"{md5_hash}.json"

            with open(json_file_path, "w") as f:
                f.write(json.dumps(doc))

            if SAVE_MARKDOWN_FILES:
                markdown_path = Path(MARKDOWN_OUTPUT_PATH)
                markdown_path.mkdir(parents=True, exist_ok=True)
                markdown_file_path = markdown_path / f"{md5_hash}.md"

                with open(markdown_file_path, "w") as f:
                    f.write(output_text)

        prompts_map_reduce = yaml.safe_load(importlib.resources.files("prompts").joinpath("map_reduce.yaml").read_text())

        task = prompts_map_reduce["task"]

        def llm_call(prompt):
            res = client.chat(
                model=MAP_REDUCE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.6,
                    "max_tokens": 20_000,
                    "num_predict": 12_000
                }
            )
            return res.message.content, res.prompt_eval_count, res.eval_count

        def llm_call_format(prompt):
            res = client.chat(
                model=MAP_REDUCE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.6,
                    "max_tokens": 20_000,
                    "num_predict": 12_000
                },
                format=Podsumowanie.model_json_schema()
            )
            return res, res.prompt_eval_count, res.eval_count

        mr = MapReduce(
            MAP_REDUCE_CONTEXT_WINDOW,
            map_prompt=prompts_map_reduce["map"],
            collapse_prompt=prompts_map_reduce["collapse"],
            reduce_prompt=prompts_map_reduce["reduce"],
            task=task,
            llm_call_map=llm_call,
            llm_call_collapse=llm_call,
            llm_call_reduce=llm_call_format
        )

        summary, total_in_tokens, total_out_tokens = mr.mapreduce(["\n\n".join(generated_texts)])
        finalne_podsumowanie = Podsumowanie.model_validate_json(summary.message.content)

        return finalne_podsumowanie.podsumowanie