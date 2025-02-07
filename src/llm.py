from functools import partial
import sys
import uuid
from typing import Optional, List
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
     RemoveMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from loguru import logger
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Dict

import openai
from retry import retry

def get_open_ai_chat_model(model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Load OpenAI chat model with environment variables."""
    load_dotenv()
    return ChatOpenAI(model=model)


# https://python.langchain.com/docs/versions/migrating_memory/
def get_memory_graph(
    model: str = "gpt-4o-mini",
    summary_prompt: Optional[str] = "Summarize the key points of the conversation.",
    system_prompt: Optional[str] = "You are a helpful assistant.",
    max_history: int = 20,
    thread_id: Optional[uuid.UUID] = None,
    debug: bool = False,  # Make debugging configurable
) -> StateGraph:
    """
    Create a memory graph for conversation management.

    Args:
        model (str): The model name to use.
        summary_prompt (Optional[str]): The prompt to use for summarizing the conversation.
        system_prompt (Optional[str]): The system prompt to use.
        max_history (int): The maximum number of messages before summarizing.
        thread_id (Optional[uuid.UUID]): The unique identifier for the conversation thread.
        debug (bool): Whether to enable debugging.

    Returns:
        StateGraph: The configured state graph for the conversation.
    """
    workflow = StateGraph(state_schema=MessagesState)
    chat_model = get_open_ai_chat_model(model)
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."
        logger.warning("System prompt not provided. Using default prompt.")
    if summary_prompt is None:
        summary_prompt = "Summarize the key points of the conversation."
        logger.warning("Summary prompt not provided. Using default prompt.")
    # Handle None cases explicitly
    _call_model = partial(
        call_model,
        model=chat_model,
        message_threshold=max_history,
        system_prompt=system_prompt,
        summary_prompt=summary_prompt,
    )

    workflow.add_edge(START, "model")
    workflow.add_node("model", _call_model)

    # Proper memory handling
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory, debug=debug)

    # Assign a unique thread ID if not provided
    thread_id = thread_id or uuid.uuid4()
    app.config = {"configurable": {"thread_id": thread_id}}

    return app

def generate_summary(
    model: ChatOpenAI, messages: List[HumanMessage], summary_prompt: str
) -> SystemMessage:
    """Generates a summary of the chat history when the message count exceeds the threshold."""
    logger.info(f"Generating summary for {len(messages)} messages using {summary_prompt}.")
    summary = model.invoke(messages + [summary_prompt])
    return summary


def get_history(state: MessagesState) -> List[HumanMessage]:
    """Retrieves the message history excluding the latest message."""
    return state["messages"][:-1]


def get_latest_human_message(state: MessagesState) -> HumanMessage:
    """Retrieves the latest human message from the state."""
    return state["messages"][-1]


def insert_system_message(
    messages: List[HumanMessage], system_message: SystemMessage
) -> List:
    """Ensures the system message is prepended to the message list."""
    return [system_message] + messages


def get_summarized_messages(
    state: MessagesState,
    model: ChatOpenAI,
    summary_prompt: str
) -> List:
    """Generates a summary message and prepares the messages for invocation."""
    message_history = get_history(state)
    summary_message = generate_summary(model, message_history, summary_prompt)
    logger.info(f"Summary: {summary_message.content}")
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
    human_message = HumanMessage(content=get_latest_human_message(state).content)
    response = model.invoke([summary_message, human_message])
    return [summary_message, human_message, response] + delete_messages


def call_model(
    state: MessagesState,
    model: ChatOpenAI,
    message_threshold: int = 4,
    system_prompt: str = "You are a helpful assistant. Answer all questions to the best of your ability. The provided chat history includes a summary of the earlier conversation.",
    summary_prompt: str = "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
) -> Dict[str, List]:
    """Calls the model with or without a summarized conversation history."""
    system_message = SystemMessage(content=system_prompt)
    message_history = get_history(state)
    number_of_messages = len(message_history)
    if number_of_messages >= message_threshold:
        logger.info(f"Summarizing {number_of_messages}>= {message_threshold} threshold messages.")
        message_updates = get_summarized_messages(state=state, model=model, summary_prompt=summary_prompt)
    else:
        logger.info(f"Calling model with {number_of_messages} messages.")
        message_updates = model.invoke(
            insert_system_message(state["messages"], system_message)
        )

    return {"messages": message_updates}


@retry(tries=-1, delay=10, backoff=5, exceptions=(openai.RateLimitError,), logger=logger)
def invoke_graph(app: StateGraph, 
                 message: str, 
                 log_tokens: bool = True) -> str:
    """Helper method to interact with the LLM and log the input/output with retries."""
    
    logger.info(f"Invoking LLM with message: {message}")

    try:
        response = app.invoke(dict(messages=[message]))
        ai_response = response["messages"][-1].content.strip()
        logger.info(f"AI Response: {ai_response}")
        
        # Log token usage
        if log_tokens:
            log_token_usage(response=response)

        return ai_response

    except openai.RateLimitError as e:
        logger.warning(f"Rate limit reached, retrying... {str(e)}")
        raise

def log_token_usage(response):
    ai = response['messages'][-1]
    if hasattr(ai, "response_metadata"):
        token_info = ai.response_metadata['token_usage']
        logger.info(f"completion_tokens: {token_info['completion_tokens']}, "
                    f"prompt_tokens: {token_info['prompt_tokens']}, "
                    f"total_tokens: {token_info['total_tokens']}")
    else:
        logger.warning("No token usage information available in response.")