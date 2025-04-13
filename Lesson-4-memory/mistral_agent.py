from langchain.agents import AgentExecutor, initialize_agent
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr
from langchain import hub
from agent_tools import tools
from langchain.memory import  ConversationBufferMemory

llm = ChatMistralAI(
    model_name= "mistral-large-latest",
    api_key= SecretStr('MjZjgUYmNJNkfpepo4w18gzuTtH3XI1G')
)

prompt = hub.pull("hwchase17/react")
agent = initialize_agent(tools, llm, )
agent_executor = AgentExecutor(agent = agent, tools = [],verbose=True, memory= memory)

response1 = agent_executor.invoke(
    {"input": "What's 3.14 divided by 0? Use available tools to calculate. Also find the latest news about NVIDIA stock."}
)
response2 = agent_executor.invoke(
    {"input": "Remind me the answer before."}
)


print(response1)
print(response2)