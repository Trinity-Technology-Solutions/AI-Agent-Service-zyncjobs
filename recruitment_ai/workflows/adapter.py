"""Workflow adapter — converts BrainState ↔ dict at the LangGraph boundary.
Nodes return BrainState internally; this adapter converts to dict for LangGraph channels.
"""
from functools import wraps
from recruitment_ai.brains.base import BrainState


def to_dict(node_fn):
    """Wrap a LangGraph node so it returns a dict for channel writes.
    The node itself receives and returns BrainState for type safety.
    """
    @wraps(node_fn)
    async def wrapped(state: BrainState) -> dict:
        result = await node_fn(state)
        if isinstance(result, BrainState):
            return result.model_dump()
        return result
    return wrapped
