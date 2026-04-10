"""
ATTONE — Control Set Manager
Analyzes control images into a consolidated tonal profile,
persists named control sets to DB, and reloads them on demand.
"""

import numpy as np
from sqlalchemy.orm import Session as DBSession
from color_profile import extract_profile
from database import ControlSet, utcnow


class ControlSetManager:

    @staticmethod
    def analyze(image_paths: list[str]) -> dict:
        if not image_paths:
            raise ValueError("No image paths provided")

        profiles = []
        for path in image_paths:
            profiles.append(extract_profile(path))

        keys = profiles[0].keys()
        averaged = {}
        for key in keys:
            values = [p[key] for p in profiles]
            averaged[key] = round(float(np.mean(values)), 2)

        return {
            "profile": averaged,
            "image_count": len(profiles),
            "individual_profiles": profiles,
        }

    @staticmethod
    def save(db: DBSession, name: str, profile_result: dict, description: str = None, source_dir: str = None) -> ControlSet:
        existing = db.query(ControlSet).filter(ControlSet.name == name).first()
        if existing:
            existing.profile_data = profile_result["profile"]
            existing.image_count = profile_result["image_count"]
            existing.description = description or existing.description
            existing.source_dir = source_dir or existing.source_dir
            existing.updated_at = utcnow()
            db.commit()
            db.refresh(existing)
            return existing

        cs = ControlSet(
            name=name,
            description=description,
            image_count=profile_result["image_count"],
            profile_data=profile_result["profile"],
            source_dir=source_dir,
        )
        db.add(cs)
        db.commit()
        db.refresh(cs)
        return cs

    @staticmethod
    def load(db: DBSession, name: str = None, cs_id: int = None) -> dict | None:
        if cs_id is not None:
            cs = db.query(ControlSet).filter(ControlSet.id == cs_id).first()
        elif name is not None:
            cs = db.query(ControlSet).filter(ControlSet.name == name).first()
        else:
            raise ValueError("Provide either name or cs_id")

        if cs is None:
            return None

        return {
            "id": cs.id,
            "name": cs.name,
            "description": cs.description,
            "image_count": cs.image_count,
            "profile": cs.profile_data,
            "source_dir": cs.source_dir,
            "created_at": cs.created_at.isoformat() if cs.created_at else None,
            "updated_at": cs.updated_at.isoformat() if cs.updated_at else None,
        }

    @staticmethod
    def list_all(db: DBSession) -> list[dict]:
        sets = db.query(ControlSet).order_by(ControlSet.created_at.desc()).all()
        return [
            {
                "id": cs.id,
                "name": cs.name,
                "description": cs.description,
                "image_count": cs.image_count,
                "created_at": cs.created_at.isoformat() if cs.created_at else None,
            }
            for cs in sets
        ]
