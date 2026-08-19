# src/pipeline/extraction.py

import csv
import json
from pathlib import Path
from typing import Dict, List
from src.core.config import paths

class EscoSkill:
    def __init__(self, skill_id: str, name: str, alt_labels: List[str]):
        self.skill_id = skill_id
        self.name = name
        self.alt_labels = alt_labels

    def to_dict(self) -> Dict:
        return {
            "id": self.skill_id,
            "preferredLabel": self.name,
            "altLabels": self.alt_labels
        }


class EscoOccupation:
    def __init__(self, occ_id: str, title: str, skill_ids: List[str]):
        self.occ_id = occ_id
        self.title = title
        self.skill_ids = skill_ids

    def to_dict(self) -> Dict:
        return {
            "id": self.occ_id,
            "preferredLabel": self.title,
            "skillIds": self.skill_ids
        }


class EscoExtractor:
    """Loads the normalized ESCO JSON files used by dataset synthesis."""

    def __init__(self, skills_path: str, occupations_path: str):
        self.skills_path = skills_path
        self.occupations_path = occupations_path
        self.skills: Dict[str, EscoSkill] = {}
        self.occupations: List[EscoOccupation] = []

    def load(self) -> None:
        with open(self.skills_path, "r", encoding="utf-8") as file:
            skills_data = json.load(file)
        with open(self.occupations_path, "r", encoding="utf-8") as file:
            occupations_data = json.load(file)

        self.skills = {
            item["id"]: EscoSkill(
                skill_id=item["id"],
                name=item["preferredLabel"],
                alt_labels=item.get("altLabels", []),
            )
            for item in skills_data
        }
        self.occupations = [
            EscoOccupation(
                occ_id=item["id"],
                title=item["preferredLabel"],
                skill_ids=item.get("skillIds", []),
            )
            for item in occupations_data
        ]

    def get_skills_for_occupation(self, occupation: EscoOccupation) -> List[EscoSkill]:
        return [
            self.skills[skill_id]
            for skill_id in occupation.skill_ids
            if skill_id in self.skills
        ]


class CsvExtractor:
    """
    Loads ESCO CSV files and converts them into JSON structures
    usable by the synthesize.py pipeline.
    """

    def __init__(self, skills_csv: str, occupations_csv: str, relations_csv: str):
        self.skills_csv = skills_csv
        self.occupations_csv = occupations_csv
        self.relations_csv = relations_csv

        self.skills: Dict[str, EscoSkill] = {}
        self.occupations: List[EscoOccupation] = []

    # -----------------------------
    # Public API
    # -----------------------------
    def load(self) -> None:
        self._load_skills_csv()
        self._load_occupations_csv()
        self._load_relations_csv()

    def export_json(self, skills_out: str, occupations_out: str) -> None:
        skills_json = [skill.to_dict() for skill in self.skills.values()]
        occupations_json = [occ.to_dict() for occ in self.occupations]

        Path(skills_out).parent.mkdir(parents=True, exist_ok=True)
        Path(occupations_out).parent.mkdir(parents=True, exist_ok=True)

        with open(skills_out, "w", encoding="utf-8") as file:
            json.dump(skills_json, file, ensure_ascii=False, indent=2)

        with open(occupations_out, "w", encoding="utf-8") as file:
            json.dump(occupations_json, file, ensure_ascii=False, indent=2)

        print(f"Exported {len(skills_json)} skills → {skills_out}")
        print(f"Exported {len(occupations_json)} occupations → {occupations_out}")

    # -----------------------------
    # Internal CSV loaders
    # -----------------------------
    def _load_skills_csv(self) -> None:
        with open(self.skills_csv, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                skill_id = row["conceptUri"]
                name = row["preferredLabel"]
                alt_labels_raw = row.get("altLabels") or ""

                alt_labels = [label.strip() for label in alt_labels_raw.splitlines() if label.strip()]

                skill = EscoSkill(skill_id=skill_id, name=name, alt_labels=alt_labels)
                self.skills[skill_id] = skill

    def _load_occupations_csv(self) -> None:
        with open(self.occupations_csv, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                occ_id = row["conceptUri"]
                title = row["preferredLabel"]

                occupation = EscoOccupation(
                    occ_id=occ_id,
                    title=title,
                    skill_ids=[],
                )
                self.occupations.append(occupation)

    def _load_relations_csv(self) -> None:
        occupations_by_id = {occupation.occ_id: occupation for occupation in self.occupations}

        with open(self.relations_csv, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                occupation = occupations_by_id.get(row["occupationUri"])
                skill_id = row["skillUri"]
                if occupation is not None and skill_id in self.skills:
                    occupation.skill_ids.append(skill_id)


def run():
    source_dir = Path("data/raw/ESCO dataset - v1.2.1 - classification - en - csv")
    extractor = CsvExtractor(
        skills_csv=str(source_dir / "skills_en.csv"),
        occupations_csv=str(source_dir / "occupations_en.csv"),
        relations_csv=str(source_dir / "occupationSkillRelations_en.csv"),
    )

    extractor.load()

    extractor.export_json(
        skills_out=paths.raw_esco_skills,
        occupations_out=paths.raw_esco_occupations
    )


if __name__ == "__main__":
    run()

