# src/core/schema.py
#
# Mirrors the response models in the consumer project
# (CV_to_Job_Guestimator, src/schemas/requirements.py and experience.py) so
# evaluation here validates exactly what instructor will enforce at serving
# time.
#
# No extra="forbid": Pydantic emits "additionalProperties": false for any
# model with it, and google-genai's response_schema converter (a restricted
# OpenAPI subset) rejects that keyword outright ("Unknown name
# additional_properties... Cannot find field"). The real consumer schemas
# only set it on JobRequirement, so plain models here match that more
# closely anyway. Field-exactness (e.g. exact requirement_id coverage) is
# checked explicitly in evaluate.py instead of relying on Pydantic for it.

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# --- Task 1: skill matcher -------------------------------------------------

class RequirementEvaluation(BaseModel):
    requirement_id: int
    matched: bool


class SkillEvaluationDecision(BaseModel):
    evaluations: List[RequirementEvaluation]


# --- Task 2: job requirements extraction ------------------------------------

class JobRequirement(BaseModel):
    skill_name: str = Field(min_length=1)


class JobRequirementsOutput(BaseModel):
    job_requirements: List[JobRequirement]

    @field_validator("job_requirements")
    @classmethod
    def validate_no_duplicates(cls, requirements: List[JobRequirement]) -> List[JobRequirement]:
        names = [r.skill_name.strip().casefold() for r in requirements]
        if len(names) != len(set(names)):
            raise ValueError("job_requirements must not contain duplicates")
        return requirements


# --- Task 3: overall work experience ----------------------------------------

class WorkRole(BaseModel):
    role_title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    match_rationale: str
    is_relevant: bool


class OverallExperienceOutput(BaseModel):
    target_job_title: str
    target_overall_years: Optional[float] = Field(default=None, ge=0.0)
    candidate_roles: List[WorkRole]


class OverallExperienceResponse(BaseModel):
    overall_experience: OverallExperienceOutput
