"""Node implementations for the reflection workflow.

This module contains the writer and reviewer node implementations that
handle response generation and review feedback.
"""

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from state import MessageResponseState, Decision, NodeName, AIReviewerResponse
from config import (
    MAX_REVISIONS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_REVISIONS_MESSAGE
)

def _create_writer_messages(
    state: MessageResponseState, feedback: Optional[str] = None
) -> list:
    """Create message list for writer node invocation.
    
    Args:
        state: Current state containing messages.
        feedback: Optional feedback from reviewer for revision.
        
    Returns:
        List of messages formatted for model invocation.

    """
    writer_prompt = open("prompts/writer_instructions.md").read()
    messages = [{"role": "system", "content": writer_prompt}] + state["messages"]
    original_message = state.get("original_customer_message", "")
    messages.append(HumanMessage(content=f"\n\nOriginal Customer Message: \n{original_message}\n\n"))
    
    if feedback:
        feedback_message = (
            f"\n\nReviewer Feedback:\n{feedback}\n\n"
            "Please revise your response based on this feedback."
        )
        messages.append(HumanMessage(content=feedback_message))
    
    return messages


def _update_writer_state(
    state: MessageResponseState, response_content: str
) -> MessageResponseState:
    """Update state after writer generates response.
    
    Args:
        state: Current state.
        response_content: Generated response content.
        
    Returns:
        Updated state dictionary.

    """
    revision_count = state.get("revision_count", 0)
    
    return {
        **state,
        "revision_count": revision_count + 1,
        "latest_message_response_by_writer": response_content,
        "messages": [
            AIMessage(content=response_content, name=NodeName.WRITER.value)
        ]
    }


def writer_node(
    state: MessageResponseState
) -> MessageResponseState:
    """Writer node that creates or revises responses to customer comments.
    
    This node generates initial responses or revises them based on reviewer
    feedback. It handles both first-time writing and revision scenarios.
    
    Args:
        state: Current workflow state containing customer comment and feedback.
        config: Runnable configuration for the node.
        
    Returns:
        Updated state with the written or revised response.

    """
    revision_count = state.get("revision_count", 0)
    latest_decision = state.get("latest_reviewer_decision")
    
    # Determine if this is a revision or initial write
    feedback = None
    if revision_count > 0 and latest_decision == Decision.REVISE.value:
        feedback = state.get("latest_feedback_for_writer", "")
    
    # Generate response
    messages = _create_writer_messages(state, feedback)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,     # Model name
        temperature=DEFAULT_TEMPERATURE
    )

    # Invoke the model with reasoning configuration
    response = llm.invoke(
        messages
    )

    # Update and return state
    return _update_writer_state(state, response.content)


def reviewer_node(state: MessageResponseState) -> MessageResponseState:
    """Reviewer node that evaluates written responses and provides feedback.
    
    This node reviews the writer's response against quality criteria and
    decides whether to approve or request revisions. Automatically approves
    after maximum revisions are reached.
    
    Args:
        state: Current workflow state containing the written response.
        
    Returns:
        Updated state with reviewer feedback and continuation decision.

    """
    revision_count = state.get("revision_count", 0)
    
    # Check if max revisions reached
    if revision_count >= MAX_REVISIONS:
        return {
            **state,
            "continue_revision": False,
            "latest_reviewer_decision": Decision.APPROVE.value,
            "messages": [
                HumanMessage(
                    content=MAX_REVISIONS_MESSAGE,
                    name=NodeName.REVIEWER.value
                ),
                HumanMessage(
                    content=f"{revision_count} reviewer decision: {Decision.APPROVE.value}", name=NodeName.REVIEWER.value
                ),
                HumanMessage(
                    content=f"{revision_count} continue revision?: {False}", name=NodeName.REVIEWER.value
                )
            ] 
        }

    # Get response and comment for review
    latest_response = state.get("latest_message_response_by_writer", "")
    original_message = state.get("original_customer_message", "")
    
    # Create review messages
    reviewer_prompt = open("prompts/reviewer_instructions.md").read()

    messages = [
        {"role": "system", "content": reviewer_prompt}
    ] + state["messages"]
    
    review_content = (
        f"\n\nOriginal Customer Message: \n{original_message}\n\n"
        f"\n\nProposed Response: \n{latest_response}\n\n"
    )
    messages.append(HumanMessage(content=review_content))
    
    # Get reviewer feedback
    # Initialize the LLM
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,     
        temperature=DEFAULT_TEMPERATURE   
    )

    response = llm.invoke(
        messages
    )
    
    feedback_text = response.content
    # Extract decision using structured output
    decision_prompt = (
        "Based on the given feedback provided by the agent, identify if "
        "the agent suggests revision or approves the content as is. "
        f"Here is the feedback: \n{feedback_text}\n\n"
    )
    reviewer_response = llm.with_structured_output(
        AIReviewerResponse
    ).invoke(decision_prompt)
    decision = reviewer_response.get("decision", Decision.REVISE.value)
    
    # Determine if another revision is needed
    continue_revision = (
        decision == Decision.REVISE.value and 
        revision_count < MAX_REVISIONS
    )

    return {
        **state,
        "latest_feedback_for_writer": feedback_text,
        "latest_reviewer_decision": decision,
        "continue_revision": continue_revision,
        "messages": [
            HumanMessage(
                content=f"{revision_count} - feedback for writer: {feedback_text}", name=NodeName.REVIEWER.value
            ),
            HumanMessage(
                content=f"{revision_count} - reviewer decision: {decision}", name=NodeName.REVIEWER.value
            ),
            HumanMessage(
                content=f"{revision_count} - continue revision?: {continue_revision}", name=NodeName.REVIEWER.value
            )
        ]
    }


def publisher_node(state: MessageResponseState) -> MessageResponseState:
    """Final node that publishes the approved response.
    
    This node is called when the response has been approved by the reviewer
    or maximum revisions have been reached. It serves as the terminal node
    before workflow completion.
    
    Args:
        state: Final workflow state containing the approved response.
        
    Returns:
        Unchanged state (ready for publication).

    """


    print("📤 Publisher node called....")
    print("✅ Final response ready to publish!")
   
    return {**state}
