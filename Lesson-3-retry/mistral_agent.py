from typing import Dict, Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr
from langchain import hub
from agent_tools import tools

llm = ChatMistralAI(
    model_name= "mistral-large-latest",
    api_key= SecretStr('MjZjgUYmNJNkfpepo4w18gzuTtH3XI1G')
)

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent = agent, tools = [],verbose=True)

def custom_executor(input_data: Dict[str, Any]):
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

response = agent_executor.invoke(
    {"input": "What's 3.14 divided by 0? Use available tools to calculate. Also find the latest news about NVIDIA stock."}
)

print(response)