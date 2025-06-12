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


# def use_short_memory(memory_step: ActionStep, agent: ToolCallingAgent):
#     agent.memory.steps = agent.memory.steps[0:2] + agent.memory.steps[:-40]


model = LiteLLMModel(
    model_id=MANAGER_MODEL,
    api_key=os.environ['OPENAI_API_KEY']
)

dataset_analyzer = ToolCallingAgent(tools=[
    GenerateNewDocument(),
    AnalyzeMetrics(),
    CreateDomainIdeas(),
],
    model=model,
    max_steps=MAX_STEPS,
    # step_callbacks=[use_short_memory],
    planning_interval=PLANNING_INTERVAL,
    prompt_templates=yaml.safe_load(importlib.resources.files("prompts").joinpath("manager.yaml").read_text()),
)

task = yaml.safe_load(importlib.resources.files("prompts").joinpath("cel.yaml").read_text())["cel"]

print(dataset_analyzer.run(task))
