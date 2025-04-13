from langchain.agents import AgentExecutor, create_react_agent
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr
from langchain import hub
from agent_tools import tools
from langchain.memory import ConversationBufferWindowMemory

llm = ChatMistralAI(
    model_name="mistral-large-latest",
    api_key=SecretStr('MjZjgUYmNJNkfpepo4w18gzuTtH3XI1G')
)

# Use ConversationBufferWindowMemory with return_messages=True
# The memory_key should match the variable expected by the prompt (usually 'chat_history')
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# Pull the standard ReAct prompt from the Hub
prompt = hub.pull("hwchase17/react-chat") # Using react-chat which is designed for memory

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor, passing the memory object
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    memory=memory,
    handle_parsing_errors=True
)

def chat_loop():
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Invoke the agent executor, it will handle memory automatically
        response = agent_executor.invoke({"input": user_input})
        print(f"Agent: {response['output']}")

if __name__ == "__main__":
    chat_loop()
