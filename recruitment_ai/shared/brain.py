"""Abstract Brain interface - all brains must implement this."""
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class BrainState(BaseModel):
    """State passed between brains in the workflow."""
    query: Optional[str] = None
    intent: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_role: Optional[str] = "candidate"
    file_content: Optional[str] = None
    file_type: Optional[str] = None
    context: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    metadata: dict = {}


class Brain(ABC):
    """Base class for all specialized brains."""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    async def run(self, state: BrainState) -> BrainState:
        """Execute the brain's main logic."""
        pass

    async def validate_input(self, state: BrainState) -> bool:
        """Validate input before processing."""
        return True

    async def post_process(self, state: BrainState) -> BrainState:
        """Post-process results."""
        return state

    async def handle_error(self, state: BrainState, error: Exception) -> BrainState:
        """Handle errors gracefully."""
        state.error = f"{self.name} error: {str(error)}"
        return state