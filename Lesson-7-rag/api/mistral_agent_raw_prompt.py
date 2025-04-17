from langchain.agents import AgentExecutor, create_react_agent
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr
from agent_tools import tools
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools.render import render_text_description

llm = ChatMistralAI(
    model_name="mistral-large-latest",
    api_key=SecretStr('MjZjgUYmNJNkfpepo4w18gzuTtH3XI1G')
)

# Use ConversationBufferWindowMemory with return_messages=True
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# == Raw Prompt Template Definition ==

# 1. Render tool descriptions for the prompt
tool_description = render_text_description(tools)
# 2. Get tool names for partial variable
tool_names = ", ".join([t.name for t in tools])

# 3. Define the prompt template string with {tool_names} placeholder
prompt_template_string = """Assistant is a large language model trained by Google.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------
Assistant has access to the following tools:

{tools}

TOOL USAGE GUIDELINES:
1. For factual questions or current events, use the 'search' tool to get up-to-date information.
2. For questions asking how to respond in a certain style, role-play, or provide creative content, use the 'retrieve_prompts' tool first to get inspiration.
3. For questions about software testing or generative AI in testing, use the 'software_testing_pdf' tool.

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

# 4. Create PromptTemplate using partial_variables for tool_names
prompt = PromptTemplate(
    input_variables=["input", "chat_history", "agent_scratchpad", "tools"], # tool_names is handled by partial_variables
    template=prompt_template_string,
    partial_variables={"tool_names": tool_names} # Provide tool_names here
)

# == Agent Creation (using the raw prompt) ==
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor, passing the memory object
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    memory=memory,
    handle_parsing_errors=True
)

