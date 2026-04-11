"""
ATTONE — Control Set Manager
Analyzes control images into a consolidated tonal profile,
detects outlier images, persists named control sets to DB,
and reloads them on demand.
"""

import os
import numpy as np
from sqlalchemy.orm import Session as DBSession
from color_profile import extract_profile
from database import ControlSet, utcnow

OUTLIER_KEYS = [
    "luminance_mean",
    "saturation_mean",
    "temperature_est_k",
    "contrast_std",
]

OUTLIER_Z_THRESHOLD = 1.8


class ControlSetManager:

    @staticmethod
    def _detect_outliers(
        profiles: list[dict], image_paths: list[str]
    ) -> list[dict]:
        if len(profiles) < 3:
            return []

        outliers = []
        flagged_indices: set[int] = set()

        for key in OUTLIER_KEYS:
            values = np.array([p[key] for p in profiles])
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std < 1e-6:
                continue

            for i, val in enumerate(values):
                z = abs(val - mean) / std
                if z >= OUTLIER_Z_THRESHOLD and i not in flagged_indices:
                    flagged_indices.add(i)
                    deviations = {}
                    for k in OUTLIER_KEYS:
                        vs = np.array([p[k] for p in profiles])
                        m = float(np.mean(vs))
                        s = float(np.std(vs))
                        if s > 1e-6:
                            deviations[k] = {
                                "value": round(float(profiles[i][k]), 2),
                                "mean": round(m, 2),
                                "z_score": round(abs(float(profiles[i][k]) - m) / s, 2),
                            }
                    outliers.append({
                        "index": i,
                        "filename": os.path.basename(image_paths[i]),
                        "path": image_paths[i],
                        "deviations": deviations,
                    })

        return outliers

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

        outliers = ControlSetManager._detect_outliers(profiles, image_paths)

        return {
            "profile": averaged,
            "image_count": len(profiles),
            "individual_profiles": profiles,
            "outliers": outliers,
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
    def delete(db: DBSession, cs_id: int) -> bool:
        cs = db.query(ControlSet).filter(ControlSet.id == cs_id).first()
        if cs is None:
            return False
        db.delete(cs)
        db.commit()
        return True

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
