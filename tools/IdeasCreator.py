import hashlib
import importlib
import json

from pathlib import Path

import numpy as np
import spacy
import yaml
from ollama import Client
from smolagents import Tool

from models.models import Token, Document
from settings import OLLAMA_URL, IDEAS_CREATOR_MODEL, JSON_OUTPUT_PATH
from utils.utils import get_dataset_metrics, format_report



class CreateDomainIdeas(Tool):
    name = "CreateDomainIdeas"
    description = 'Użyj tego narzędzia do wygenerowania listy domen. Możesz się nimi zainspirować podczas generowania tekstu.'

    inputs = {}

    output_type = "string"

    def forward(self,):
        # iteracja po istniejących dokumentach
        docs = []
        for file in Path(JSON_OUTPUT_PATH).glob("*.json"):
            with open(file, "r", encoding="utf8") as f:
                data = json.loads(f.read())
                doc = Document(
                    text=data["text"],
                    domain=data["domain"],
                    tokens=[Token.model_validate_json(t) for t in data["tokens"]],
                    sentences=[[Token.model_validate_json(t) for t in sentence] for sentence in data["sentences"]],
                    pisarek_index=data["pisarek_index"],
                    style=data["style"],
                    target_audience=data["target_audience"],
                )
                docs.append(doc)

        # stworzenie zbioru domen, które ją są w użyciu
        already_used_domains = set([doc.domain for doc in docs])
        if len(already_used_domains) > 0:
            already_used_domains_string = '\n'.join(already_used_domains)
        else:
            already_used_domains_string = '-BRAK DANYCH-'

        client = Client(
            host=OLLAMA_URL
        )

        prompts = yaml.safe_load(importlib.resources.files("prompts").joinpath("ideas_creator.yaml").read_text())

        response = client.chat(
            model=IDEAS_CREATOR_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompts["system"]
                },
                {
                    "role": "user",
                    "content": prompts["user"].format(already_used_domains_string=already_used_domains_string)
                },
            ],
        )

        res_content = response["message"]["content"]

        # rozdzielenie początkowych pomysłów od tych najlepszych
        if "Najlepsze pomysły:" in res_content:
            odpowiedz = res_content.split("Najlepsze pomysły:")[1]
        else:
            odpowiedz = res_content


        return odpowiedz
