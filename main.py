import importlib
import os

import yaml
from dotenv import load_dotenv
from smolagents import LiteLLMModel, ToolCallingAgent

from settings import MANAGER_MODEL, MAX_STEPS, PLANNING_INTERVAL
from tools.AnalyzeMetrics import AnalyzeMetrics
from tools.GenerateText import GenerateNewDocument
from tools.IdeasCreator import CreateDomainIdeas

load_dotenv()

# kod poniżej ma za zadanie usuwać historię wiadomości agenta
# przekłada się to na ilość przetwarzanych tokenów z każdą wiadomością i pomoże nie przekroczyć okna kontekstowego
# minusem tego rozwiązania jest gubienie trochę kontekstu przez model agenta
# użycie jest dość eksperymentalne i powinno być dobrze przetestowane

# def use_short_memory(memory_step: ActionStep, agent: ToolCallingAgent):
#     agent.memory.steps = agent.memory.steps[0:2] + agent.memory.steps[:-40]


# Zdefiniuj model managera
model = LiteLLMModel(
    model_id=MANAGER_MODEL,
    api_key=os.environ['OPENAI_API_KEY']
)

# Przygotuj narzędzia oraz inne ustawienia
dataset_analyzer = ToolCallingAgent(tools=[
    GenerateNewDocument(),
    AnalyzeMetrics(),
    CreateDomainIdeas(),
],
    model=model,
    max_steps=MAX_STEPS, # maksymalna ilość kroków działania dla Managera
    # step_callbacks=[use_short_memory],
    planning_interval=PLANNING_INTERVAL, # co ile kroków Manager powinien odświeżyć plan działania
    prompt_templates=yaml.safe_load(importlib.resources.files("prompts").joinpath("manager.yaml").read_text()), # definicja promptów managera
)

# prompt celu do którego ma dążyć manager
task = yaml.safe_load(importlib.resources.files("prompts").joinpath("cel.yaml").read_text())["cel"]

# uruchomienie procesu
print(dataset_analyzer.run(task))
