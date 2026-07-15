"""Authenticate node — verifies user identity is present in BrainState.
JWT verification happens in FastAPI before the workflow is invoked.
This node is a guard: if user_id is missing, short-circuit with an error.
"""
from recruitment_ai.brains.base import BrainState


async def authenticate_node(state: BrainState) -> BrainState:
    if not state.user_id:
        state.error = "Authentication required"
    return state
