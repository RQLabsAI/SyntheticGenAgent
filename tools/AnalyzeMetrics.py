import importlib
import json
from pathlib import Path

import yaml
from litellm import completion
from smolagents import Tool

from models.models import Document, Token
from settings import ANALYST_MODEL
from utils.utils import get_dataset_metrics, format_report


def ask_reasoning_model(question: str, report: str) -> str:
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
        docs = []
        for file in Path("./generateddata").glob("*.json"):
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

        if len(docs) > 0:
            metrics = get_dataset_metrics(docs)

            return ask_reasoning_model(
                question,
                format_report(metrics)
            ) + f"\n\n Aktualnie w zbiorze danych znajduje się {len(docs)} dokumentów."
        return ask_reasoning_model(
            question,
            "Zbiór danych jest pusty. Metryki nie zostały wyliczone."
        )
