---
title: 'Running State: Enhancing Short-Term Memory in AI Agents'
date: 2024-06-30
categories:
  - Agents
  - LLM
tags:
  - Intermediate
draft: false
---

# Running State: Enhancing Short-Term Memory in AI Agents

In the rapidly evolving field of artificial intelligence, chat agents have become integral to customer service, technical support, online sales, and more. These agents solve complex tasks by breaking them down into smaller, manageable sub-tasks, and utilizing tools to retrieve knowledge and perform actions. Additionally, they memorize the context of conversations to make informed decisions.

<!-- more -->

Typically, these agents comprise three key modules: `tools`, `planning`, and `memory`. The `tools` module retrieves knowledge and performs actions, the `planning` module decomposes complex tasks into simpler sub-tasks, and the `memory` module stores and retrieves conversational context.

In this post, we will delve into the concept of running state, a technique for managing the short-term memory of AI agents. We will examine how a running state can enhance agent performance across various tasks and provide examples of its implementation.

## Prerequisites

To follow the code examples, ensure you have Python 3.12 installed. Additionally, you will need the following libraries:

- [LangChain](https://python.langchain.com/) - A Python library for building conversational AI agents.
- [Pydantic](https://pydantic-docs.helpmanual.io/) - A data validation and settings management library.
- [Streamlit](https://streamlit.io/) - An open-source framwork for quickly building web applications.

You can find the complete code in the [examples-agents repository](https://github.com/maltehedderich/examples-agents).

## The Memory Module

The memory module in an AI agent is crucial for storing and retrieving conversational context. It maintains information about the ongoing conversation, including user queries, agent responses, and task-related data. This context is essential for the agent to understand user intent, provide relevant responses, and maintain coherence throughout the interaction.

The memory module typically comprises two components: `long-term memory` and `short-term memory`.

- **Long-term memory** stores persistent information relevant across multiple conversations, such as user preferences, past converstations, and domain-specific knowledge.
- **Short-term memory** holds transient information that is related to the current conversation, such as the current user query, system state, and intermediate results.

A basic approach to managing short-term memory is maintaining the conversation history as a sequence of dialogue turns. However, this method does not scale well. The memory requirements grow linearly with the conversation length, potentially exceeding the maximum context length in terms of tokens for the Large Language Model (LLM) in use. Even if the LLM can handle a large context window, the quality of responses may degrade as the context becomes more diluted. Additionally, each extra token increases the cost of inference.

## Introducing Running State

Running state is a technique designed to overcome the limitations of short-term memory management in the form of conversation history in AI agents. It maintains a compact, structured representation of the conversation context, capturing only the essential information needed for decision-making. This state is dynamically updated as the conversation progresses, ensuring the agent has access to relevant context without retaining unnecessary details.

In the following sections, we will outline the key steps involved in implementing running state in an AI agent.

### 1. Knowledge Base Definition

The first step in implementing a running state is to identify the most relevant information from the conversation history that needs to be retained. This may include:

- General information about the user
- A summary of the conversation history
- Key entities or topics
- Unresolved problems
- The current goal or task

Additionally, you may want to include information specific to the general domain of your agent.

These are then defined using a structured representation, typically as a Pydantic `BaseModel`, which facilitates seamless integration with LangChain.

[agent/memory.py](https://github.com/maltehedderich/examples-agents/blob/50ee37de343604b903e50f59e9bf2c222af519cf/langchain-example/langchain_example/agent/memory.py)

```python
from pydantic import BaseModel, Field
from langchain_example.services.jira import JiraIssue

class KnowledgeBase(BaseModel):
    # General information about the user and the conversation.
    first_name: str = Field("unknown", description="The first name of the user which is used to personalize the responses.")
    last_name: str = Field("unknown", description="The last name of the user which is used to personalize the responses.")

    discussion_summary: str = Field("unknown", description="Summary of the discussion including points discussed and possible action items.")
    open_problems: list[str] = Field([], description="Topics of the conversation that are still not resolved.")
    current_goals: list[str] = Field([], description="Current goal of the agent to address.")

    # Domain-specific information.
    jira_issue: JiraIssue | None = None
```

We named our running state `KnowledgeBase` and utilized the Pydantic `Field` class to add additional information to the fields. LangChain will use this metadata to provide explanations about the fields to the LLM.

While general information is common across most agents, domain-specific details will differ based on the use case. For instance, a travel agent might include fields for flight details, hotel reservations, and travel preferences. In contrast, an e-commerce agent might have fields for shopping cart contents, product recommendations, and order status.

### 2. Update Mechanism - Internal Chain

To enhance the robustness of your agent, it's advisable to separate the state update mechanism from the main operation of the agent. This separation allows the agent to focus on its primary task, reducing the risk of unintended side effects.

In LangChain, the chain that the user interacts with is typically referred to as the `external` chain. Conversely, `internal` chains operate in the background to support the agent.

In the following example, we will define an internal chain that updates the running state based on the latest user input and agent response. This internal chain will be executed after each run of the actual agent chain.

[agent/memory.py](https://github.com/maltehedderich/examples-agents/blob/50ee37de343604b903e50f59e9bf2c222af519cf/langchain-example/langchain_example/agent/memory.py)

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_example.settings import settings
from langchain_openai import AzureChatOpenAI

# 1. Define output parser as it's also used for format_instructions
knowledge_parser = PydanticOutputParser(pydantic_object=KnowledgeBase)

# 2. Define prompt template for internal update chain
update_knowledge_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=(
                    "You are an assistant for a chat agent."
                    " Your task is to update the knowledge base with the latest information."
                    "\n\n{format_instructions}"
                ),
                input_variables=[],
                partial_variables={"format_instructions": knowledge_parser.get_format_instructions()},
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=(
                    "OLD KNOWLEDGE BASE: {knowledge_base}\n\n"
                    "USER MESSAGE: {input}\n\n"
                    "ASSISTANT RESPONSE: {output}\n\n"
                    "NEW KNOWLEDGE BASE: "
                ),
                input_variables=["knowledge_base", "input", "output"],
            )
        ),
    ]
)

# 3. Define chat model for internal update chain
llm = AzureChatOpenAI(
    azure_endpoint=str(settings.azure_openai_endpoint),
    model=settings.azure_openai_deployment_name,
    openai_api_version=settings.azure_openai_api_version,
    api_key=settings.azure_openai_api_key.get_secret_value(),
    model_kwargs={"response_format": {"type": "json_object"}},
    temperature=0,
)

# 4. Create chain from prompt, llm and parser
update_knowledge_chain = update_knowledge_prompt | llm | knowledge_parser
```

First, we created an output parser to parse the output of the LLM into a `KnowledgeBase` object. LangChain provides a convenient way to define the parser using the `PydanticOutputParser` class, which also supplies the format instructions to include in the prompt template.

Next, we created a `ChatPromptTemplate` to instruct the LLM to update the knowledge base with the latest information. In addition to the usual `input_variables`, we utilized `partial_variables` to pass the format instructions to the prompt template. Partial variables allow us to bind variables to the prompt template, eliminating the need to specify them during the actual invocation of the prompt.

Finally, we created a `AzureChatOpenAI` that uses the Azure OpenAI API and combined the prompt, LLM, and parser into a chain. We used a Pydantic `BaseSettings` object to access the Azure OpenAI API credentials. You can find the full definition of this object in the `settings.py` file in the [examples-agents repository](https://github.com/maltehedderich/examples-agents/blob/50ee37de343604b903e50f59e9bf2c222af519cf/langchain-example/langchain_example/settings.py).

Depending on the use case, you may want to adjust the update mechanism to capture additional context or only update every few conversation turns.

### 3. Integrating the Knowledge Base into the Agent Context - External Chain

In the external chain, the knowledge base extends the context provided to the LLM by incorporating it in the prompt template.

[agent/core.py](https://github.com/maltehedderich/examples-agents/blob/50ee37de343604b903e50f59e9bf2c222af519cf/langchain-example/langchain_example/agent/core.py)

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_example.agent.tools import all_tools
from langchain_example.settings import settings
from langchain_openai import AzureChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful assistant. All your answers are concise and to the point."
                " Your running knowledge base is {knowledge_base}."
                " This is ONLY for your reference; do not share this with the user."
            ),
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = AzureChatOpenAI(
    azure_endpoint=str(settings.azure_openai_endpoint),
    model=settings.azure_openai_deployment_name,
    openai_api_version=settings.azure_openai_api_version,
    api_key=settings.azure_openai_api_key.get_secret_value(),
    temperature=0,
)

agent = create_tool_calling_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)
```

In our system prompt, we incorporated a placeholder for the `knowledge_base`. Additionally, we specified that this information is solely for the agent's reference and should not be disclosed to the user.

We also included a `chat_history` placeholder to pass parts of the conversation history to the LLM. By using only the most recent conversation turns, we maintain a manageable context size and prevent the LLM from being distracted by irrelevant information. The running state, represented by our knowledge base, ensures that relevant context from previous interactions is preserved and utilized for informed decision-making.

### 4. Running the Agent

In our example, we utilized Streamlit along with its session state feature. This approach allows us to maintain the agent's running state across various user interactions. Note that the implementation details differ depending on the framework you choose.

[main.py](https://github.com/maltehedderich/examples-agents/blob/50ee37de343604b903e50f59e9bf2c222af519cf/langchain-example/langchain_example/main.py)

```python
from typing import Literal

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_example.agent.core import agent_executor
from langchain_example.agent.memory import KnowledgeBase, update_knowledge_chain
from langchain_example.settings import settings


def initailize_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = ChatMessageHistory()
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = KnowledgeBase()


def display_messages() -> None:
    ...


def display_and_append_message(role: Literal["assistant", "user"], content: str) -> None:
    ...


def chat() -> None:
    display_messages()
    if prompt := st.chat_input("How can I help you?"):
        display_and_append_message("user", prompt)

        # Invoke the agent to generate a response
        response = agent_executor.invoke(
            {
                "input": prompt,
                "chat_history": st.session_state.history.messages[-4:], # Only the last 4 messages
                "knowledge_base": st.session_state.knowledge_base,
            }
        )
        display_and_append_message("assistant", response["output"])

        # Update the knowledge base
        st.session_state.knowledge_base = update_knowledge_chain.invoke(
            {
                "knowledge_base": st.session_state.knowledge_base,
                "input": prompt,
                "output": response["output"],
            }
        )


def main() -> None:
    st.set_page_config(page_title=settings.app_name, page_icon="🤖")
    st.title(settings.app_name)
    initailize_session()
    chat()
```

We initialize the session state with the chat history and the knowledge base. We use an empty `ChatMessageHistory` object to store the conversation history and an empty `KnowledgeBase` object to maintain the running state. In your application, you may want to persist the chat history by storing it in a database for long-term retention. Additionally, you might want to pre-populate the knowledge base with initial information, such as the user's name from your authorization system.

We have omitted the display functions as they are not relevant to the running state concept. The `chat` function serves as the main entry point for the conversation. It displays the chat history, prompts the user for input, and calls the agent to generate a response. The response is then shown to the user, and the knowledge base is updated with the latest input and output. Since the agent is invoked with the knowledge base from the session state, it will always operate on the most recent version of the running state.

## Conclusion

In this post, we explored the concept of running state as a method for managing the short-term memory of AI agents. By maintaining a concise and structured representation of the conversation context, agents can make informed decisions without retaining superfluous details. We covered the essential steps for implementing running state, which include defining the knowledge base, updating it with an additional internal chain, and integrating it into the agent's context.

If you are interested in learning more about this concept, we recommend NVIDIA's course [Building RAG Agents with LLMs](https://resources.nvidia.com/en-us-ai-large-language-models/building-rag-agents-with-llms-dli-course?ncid=no-ncid).
