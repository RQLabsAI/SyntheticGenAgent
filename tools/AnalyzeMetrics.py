import importlib
import json
from pathlib import Path

import yaml
from litellm import completion
from smolagents import Tool

from models.models import Document, Token
from settings import ANALYST_MODEL, JSON_OUTPUT_PATH
from utils.utils import get_dataset_metrics, format_report


def ask_reasoning_model(question: str, report: str) -> str:
    """
    Odpytywanie modelu wnioskującego. Przyjmuje na wejściu pytanie oraz wygenerowany raport metryk.

    :param question: pytanie na jakie model ma odpowiedzieć
    :param report: wygenerowany raport
    :return: odpowiedź na pytanie
    """
    prompts = yaml.safe_load(importlib.resources.files("prompts").joinpath("data_scientist.yaml").read_text())

    response = completion(
        model=ANALYST_MODEL,
        messages=[
            {
                "role": "system",
                "content": prompts["system"]
            },
            {
                "role": "user",
                "content": prompts["user"].format(report=report, question=question)
            }
        ],
    )

    output_text = response.choices[0].message.content

    return output_text


class AnalyzeMetrics(Tool):
    name = "AnalyzeMetrics"
    description = ("To narzędzie pozwala ci dowiedzieć się czegoś o obecnym stanie zbioru danych. Możesz uzyskać odpowiedź na dowolne "
                   "pytanie związane z ze zbiorem danych, zapytać się o rekomendacje, czy też potencjalne pomysły na poprawę jakości, "
                   "czy też stopnia zdywersyfikowania zbioru danych. Narzędzie to jest bardzo wykwalifikowane w kwestiach związanych "
                   "z analizą zbioru danych.")

    inputs = {
        "question": {
            "type": "string",
            "description": "Pojedyńcze pytanie lub kilka związane z obecnym stanem zbioru danych, na które potrzebujesz odpowiedzi. Najoptymalniejszą opcją jest zadać wiele pytań na raz.",
        }
    }
    output_type = "string"

    def forward(self, question):
        # w pierwszym kroku zbierz wszystkie dokumenty w formacie JSON
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

        # jeżeli zbiór danych nie jest pusty, wygeneruj raport
        if len(docs) > 0:
            metrics = get_dataset_metrics(docs)

            return ask_reasoning_model(
                question,
                format_report(metrics)
            ) + f"\n\n Aktualnie w zbiorze danych znajduje się {len(docs)} dokumentów."
        # jeżeli zbiór jest pusty, niech model  odpowie, że jest to zbiór pusty
        # w zasadzie można by też zwrócić od razu odpowiedź, że zbiór jest pusty,
        # ale zostawiłem to, by odpowiedź była lepiej przygotowana do konkretnego
        # pytania
        return ask_reasoning_model(
            question,
            "Zbiór danych jest pusty. Metryki nie zostały wyliczone."
        )
