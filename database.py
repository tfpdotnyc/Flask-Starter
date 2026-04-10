"""
ATTONE — Database Models (SQLAlchemy + SQLite)
All models from the ATTONE PRD data model.
"""

import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DB_PATH = os.path.join(os.path.dirname(__file__), "attone.db")
DATABASE_URL = "sqlite:///%s" % DB_PATH

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class ControlSet(Base):
    __tablename__ = "control_sets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    image_count = Column(Integer, default=0)
    profile_data = Column(JSON, nullable=True)
    source_dir = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    sessions = relationship("Session", back_populates="control_set")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    control_set_id = Column(Integer, ForeignKey("control_sets.id"), nullable=False)
    status = Column(String(50), default="pending")
    input_dir = Column(String(500), nullable=True)
    output_dir = Column(String(500), nullable=True)
    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    completed_at = Column(DateTime, nullable=True)

    control_set = relationship("ControlSet", back_populates="sessions")
    clusters = relationship("Cluster", back_populates="session")
    images = relationship("SessionImage", back_populates="session")
    export_config = relationship("ExportConfig", back_populates="session", uselist=False)


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    label = Column(String(100), nullable=True)
    centroid_data = Column(JSON, nullable=True)
    image_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)

    session = relationship("Session", back_populates="clusters")
    images = relationship("SessionImage", back_populates="cluster")


class SessionImage(Base):
    __tablename__ = "session_images"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)
    filename = Column(String(500), nullable=False)
    original_path = Column(String(500), nullable=False)
    toned_path = Column(String(500), nullable=True)
    status = Column(String(50), default="pending")
    error_message = Column(Text, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    format = Column(String(50), nullable=True)
    profile_data = Column(JSON, nullable=True)
    delta_data = Column(JSON, nullable=True)
    correction_strength = Column(Float, default=1.0)
    skin_tone_clamped = Column(Boolean, default=False)
    processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    processed_at = Column(DateTime, nullable=True)

    session = relationship("Session", back_populates="images")
    cluster = relationship("Cluster", back_populates="images")
    edit_stubs = relationship("EditStub", back_populates="image")


class ExportConfig(Base):
    __tablename__ = "export_configs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, unique=True)
    output_format = Column(String(20), default="JPEG")
    quality = Column(Integer, default=95)
    output_dir = Column(String(500), nullable=True)
    naming_template = Column(String(255), default="{original_name}_toned")
    color_space = Column(String(20), default="sRGB")
    resize_max_px = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    session = relationship("Session", back_populates="export_config")


class EditStub(Base):
    __tablename__ = "edit_stubs"

    id = Column(Integer, primary_key=True, index=True)
    session_image_id = Column(Integer, ForeignKey("session_images.id"), nullable=False)
    edit_type = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=True)
    applied = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    image = relationship("SessionImage", back_populates="edit_stubs")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
