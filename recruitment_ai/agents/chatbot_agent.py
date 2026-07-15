"""Chatbot agent — general conversation, career advice, interview prep.
Wraps ChatbotBrain, CareerBrain from brains.chatbot and brains.candidate.
"""
from recruitment_ai.brains.chatbot.chatbot_brain import chatbot_brain
from recruitment_ai.brains.candidate.career_brain import career_brain

chatbot_agent = {
    "chat": chatbot_brain,
    "career": career_brain,
}
