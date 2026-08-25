# src/parser.py

import csv
import json
from typing import List, Dict, Any
from src.core.schema import SkillEvaluationDecision


class CsvParser:
    """Parses ESCO CSV files into Python objects."""

    @staticmethod
    def load_csv(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)


class JsonParser:
    """Handles JSON validation and serialization using Pydantic."""

    @staticmethod
    def validate_json(text: str) -> bool:
        try:
            obj = json.loads(text)
            SkillEvaluationDecision.model_validate(obj)
            return True
        except Exception:
            return False

    @staticmethod
    def parse_json(text: str) -> SkillEvaluationDecision:
        obj = json.loads(text)
        return SkillEvaluationDecision.model_validate(obj)

    @staticmethod
    def to_json(model: SkillEvaluationDecision) -> str:
        return model.model_dump_json(indent=2)
