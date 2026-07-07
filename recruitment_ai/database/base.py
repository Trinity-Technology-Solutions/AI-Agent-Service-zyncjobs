"""SQLAlchemy declarative base - separate from engine init."""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
