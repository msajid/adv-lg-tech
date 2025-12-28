"""Customer Comment Response System using LangGraph Reflection Pattern.

This application demonstrates a reflection pattern where a Writer node creates
initial responses to customer comments, and a Reviewer node provides feedback
to improve the responses before finalizing them.
"""

from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables FIRST before importing other modules
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from state import MessageResponseState, Decision, NodeName
from nodes import writer_node, reviewer_node, publisher_node



def should_continue(state: MessageResponseState) -> str:
    """Conditional edge determining the next node in the workflow.
    
    Decides whether to route back to the writer for revisions or proceed
    to the publisher for final approval.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Name of the next node to execute (writer_node or publisher_node).

    """
    if state.get("continue_revision", False):
        return NodeName.WRITER.value
    return NodeName.PUBLISHER.value
    

def create_reflection_graph() -> StateGraph:
    """Create and configure the LangGraph reflection workflow.
    
    Constructs a state graph with three nodes (writer, reviewer, publisher)
    connected in a reflection pattern where the reviewer can send responses
    back to the writer for revision.
    
    Returns:
        Configured StateGraph ready for compilation.

    """
    workflow = StateGraph(MessageResponseState)
    
    # Add nodes
    workflow.add_node(NodeName.WRITER.value, writer_node)
    workflow.add_node(NodeName.REVIEWER.value, reviewer_node)
    workflow.add_node(NodeName.PUBLISHER.value, publisher_node)
    
    # Add edges
    workflow.add_edge(START, NodeName.WRITER.value)
    workflow.add_edge(NodeName.WRITER.value, NodeName.REVIEWER.value)
    workflow.add_conditional_edges(
        NodeName.REVIEWER.value,
        should_continue,
        {
            NodeName.WRITER.value: NodeName.WRITER.value,
            NodeName.PUBLISHER.value: NodeName.PUBLISHER.value
        }
    )
    workflow.add_edge(NodeName.PUBLISHER.value, END)
    return workflow


async def process_customer_message(
    customer_message: str,
    thread_id: str = "message_112233",
    checkpointer: Optional[InMemorySaver] = None
) -> Dict[str, Any]:
    """Process a customer comment through the reflection workflow.
    
    Executes the complete reflection pattern: initial response generation,
    review, potential revisions, and final approval.
    
    Args:
        customer_comment: The customer's feedback text to respond to.
        thread_id: Unique identifier for this conversation thread.
        user_type: Type of user (guest or member).
        checkpointer: Optional checkpointer for state persistence and history.
        
    Returns:
        Final state dictionary containing the approved response and metadata.

    """
    workflow = create_reflection_graph()
    
    app = (
        workflow.compile(checkpointer=checkpointer)
        if checkpointer
        else workflow.compile()
    )

    # Initialize workflow state
    initial_state = {
        "messages": [HumanMessage(content=customer_message)],
        "original_customer_message": customer_message,
        "revision_count": 0,
        "latest_feedback_for_writer": "",
        "latest_message_response_by_writer": "",
        "latest_reviewer_decision": Decision.REVISE.value,
        "continue_revision": True
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    final_state = await app.ainvoke(
        initial_state,
        config=config
    )
    
    return final_state


async def main() -> None:
    """Demonstrate the reflection pattern with a sample customer comment."""
    print("🔄 Customer Comment Response System - Reflection Pattern Demo\n")
    
    # Sample customer comment
    sample_comment = (
        "I absolutely love my new laptop! The battery life is amazing and "
        "it's so fast. Best purchase I've made this year!"
    )
    
    # Process the comment
    checkpointer = InMemorySaver()
    result = await process_customer_message(
        customer_message=sample_comment,
        checkpointer=checkpointer,
        thread_id="message_112233"
    )
    
    # Display results
    print(f"\n📝 REVISION COUNT: {result.get('revision_count', 0)}")
    print(f"\n✅ FINAL RESPONSE:")
    print("─" * 80)
    print(result.get('latest_message_response_by_writer', 'N/A'))
    print("─" * 80)
    
    latest_feedback = result.get('latest_feedback_for_writer', '')
    if latest_feedback:
        print(f"\n📋 FINAL REVIEWER FEEDBACK:")
        print("─" * 80)
        print(latest_feedback)
        print("─" * 80)
    
    
    print("\n✨ Status: Response ready to send!")
    print("\n🎉 Demo completed! All customer comments have been processed.\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())