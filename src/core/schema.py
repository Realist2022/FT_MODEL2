# src/schema.py

from pydantic import BaseModel
from typing import List

class JobRequirement(BaseModel):
    skill_name: str

class SkillsEvaluation(BaseModel):
    requirement_category: str
    job_requirements: List[JobRequirement]
    matched_cv_skills: List[str]
    missing_cv_skills: List[str]
    rationale: str

class SkillsEvaluationWrapper(BaseModel):
    skills_evaluation: SkillsEvaluation
