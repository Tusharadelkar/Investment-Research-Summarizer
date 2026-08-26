import json
import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, String, Float, Boolean, Text, Integer, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from flask_login import UserMixin

Base = declarative_base()

class UserModel(UserMixin, Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    api_key = Column(String, nullable=True)  # User's personal LLM API key

    documents = relationship("DocumentModel", back_populates="user")


class DocumentModel(Base):
    __tablename__ = 'documents'

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    
    user = relationship("UserModel", back_populates="documents")
    
    elapsed_seconds = Column(Float, nullable=False)
    llm_available = Column(Boolean, nullable=False)
    vision_available = Column(Boolean, nullable=False)
    visual_pages = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analytics = relationship("DocumentAnalyticsModel", back_populates="document", uselist=False, cascade="all, delete-orphan")
    overview = relationship("DocumentOverviewModel", back_populates="document", uselist=False, cascade="all, delete-orphan")

    def set_visual_pages(self, value):
        self.visual_pages = json.dumps(value)
        
    def get_visual_pages(self):
        return json.loads(self.visual_pages) if self.visual_pages else []

    def public_payload(self):
        """
        Filters and structures the raw document data to return a clean payload
        for the web interface.
        """
        return {
            "document_id": self.id,
            "filename": self.filename,
            "analytics": {
                "pages": self.analytics.pages,
                "words": self.analytics.words,
                "characters": self.analytics.characters,
                "chunks": self.analytics.chunks,
                "images": self.analytics.images,
                "top_terms": json.loads(self.analytics.top_terms) if self.analytics.top_terms else [],
                "topics": json.loads(self.analytics.topics) if self.analytics.topics else [],
                "financial_metrics": json.loads(self.analytics.financial_metrics) if self.analytics.financial_metrics else [],
                "page_distribution": json.loads(self.analytics.page_distribution) if self.analytics.page_distribution else [],
            } if self.analytics else {},
            "overview": {
                "summary": self.overview.summary,
                "key_findings": self.overview.get_json_field("key_findings"),
                "risks": self.overview.get_json_field("risks"),
                "opportunities": self.overview.get_json_field("opportunities"),
                "note": self.overview.note
            } if self.overview else {},
            "processing_seconds": round(self.elapsed_seconds, 2),
            "llm_available": self.llm_available,
            "vision_available": self.vision_available,
            "visual_pages": [
                {
                    "page": page,
                    "url": f"/api/documents/{self.id}/pages/{page}/preview",
                }
                for page in self.get_visual_pages()[:18]
            ],
        }


class DocumentAnalyticsModel(Base):
    __tablename__ = 'document_analytics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    pages = Column(Integer, nullable=False)
    words = Column(Integer, nullable=False)
    characters = Column(Integer, nullable=False)
    chunks = Column(Integer, nullable=False)
    images = Column(Integer, nullable=False)
    top_terms = Column(Text, nullable=False)
    topics = Column(Text, nullable=False)
    financial_metrics = Column(Text, nullable=False)
    page_distribution = Column(Text, nullable=False)
    
    document = relationship("DocumentModel", back_populates="analytics")


class DocumentOverviewModel(Base):
    __tablename__ = 'document_overview'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    summary = Column(Text, nullable=False)
    key_findings = Column(Text, nullable=False) # JSON
    risks = Column(Text, nullable=False) # JSON
    opportunities = Column(Text, nullable=False) # JSON
    note = Column(Text, nullable=True)
    
    document = relationship("DocumentModel", back_populates="overview")
    
    def set_json_field(self, field_name, value):
        setattr(self, field_name, json.dumps(value))
        
    def get_json_field(self, field_name):
        val = getattr(self, field_name)
        return json.loads(val) if val else []

# Determine path for the sqlite database
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "documents.db"

# Initialize SQLAlchemy Engine
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
