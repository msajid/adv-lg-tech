"""State and data structures for the reflection workflow."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from langgraph.graph import MessagesState


class Decision(str, Enum):
    """Enumeration for reviewer decision types."""

    APPROVE = "APPROVE"
    REVISE = "REVISE"


class NodeName(str, Enum):
    """Enumeration for node names in the graph."""

    WRITER = "writer_node"
    REVIEWER = "reviewer_node"
    PUBLISHER = "publisher_node"


@dataclass
class AIReviewerResponse:
    """Represents the AI reviewer's decision.
    
    Attributes:
        decision: The reviewer's decision (APPROVE or REVISE).

    """

    decision: Literal["APPROVE", "REVISE"]


class MessageResponseState(MessagesState):
    """Extended state for tracking reflection workflow.
    
    Attributes:
        revision_count: Number of revisions performed.
        original_customer_comment: The original customer feedback text.
        latest_feedback_for_writer: Most recent feedback from reviewer.
        latest_message_response_by_writer: Most recent response written.
        latest_reviewer_decision: Current decision from reviewer (APPROVE/REVISE).
        continue_revision: Flag indicating whether to continue revisions.

    """

    revision_count: int = 0
    original_customer_message: str
    latest_feedback_for_writer: str
    latest_message_response_by_writer: str
    latest_reviewer_decision: str
    continue_revision: bool = True
