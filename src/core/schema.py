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
    requirement_id: int = Field(description="The supplied numeric requirement ID.")
    matched: bool = Field(description="Whether the CV satisfies this requirement.")


class SkillEvaluationDecision(BaseModel):
    evaluations: List[RequirementEvaluation] = Field(
        description="One match decision for every supplied requirement ID."
    )


# --- Task 2: job requirements extraction ------------------------------------

class JobRequirement(BaseModel):
    skill_name: str = Field(
        min_length=1,
        description="One atomic technical or operational skill_name.",
    )


class JobRequirementsOutput(BaseModel):
    job_requirements: List[JobRequirement] = Field(
        description="Unique atomic capabilities from the job description."
    )

    @field_validator("job_requirements")
    @classmethod
    def validate_no_duplicates(cls, requirements: List[JobRequirement]) -> List[JobRequirement]:
        names = [r.skill_name.strip().casefold() for r in requirements]
        if len(names) != len(set(names)):
            raise ValueError("job_requirements must not contain duplicates")
        return requirements


# --- Task 3: overall work experience ----------------------------------------

class WorkRole(BaseModel):
    role_title: str = Field(description="Title of the candidate's position.")
    start_date: Optional[str] = Field(
        default=None,
        description="Role start date in YYYY-MM format, or null when missing.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Role end date in YYYY-MM format, Present, or null when missing.",
    )
    match_rationale: str = Field(
        description="Brief comparison of this role with the target job."
    )
    is_relevant: bool = Field(
        description="Whether the role provides directly relevant target-job experience."
    )


class OverallExperienceOutput(BaseModel):
    target_job_title: str = Field(description="Target role title from the listing.")
    target_overall_years: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Explicit overall experience required, or null when unspecified.",
    )
    candidate_roles: List[WorkRole] = Field(
        description="Professional roles supported by the candidate CV."
    )


class OverallExperienceResponse(BaseModel):
    overall_experience: OverallExperienceOutput = Field(
        description="Extracted overall target and candidate role experience."
    )
