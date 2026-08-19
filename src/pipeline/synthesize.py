# src/synthesize.py

import json
import random
from typing import List, Dict
from src.core.config import paths
from src.pipeline.extraction import EscoExtractor, EscoOccupation, EscoSkill


class SkillsExample:
    """
    Represents one training example:
    - instruction: text prompt (JD + CV)
    - output: target JSON (skills_evaluation)
    """

    def __init__(self, instruction: str, output: Dict):
        self.instruction = instruction
        self.output = output

    def to_jsonl(self) -> str:
        return json.dumps({
            "instruction": self.instruction,
            "output": self.output
        }, ensure_ascii=False)


class DatasetSynthesizer:
    def __init__(self, extractor: EscoExtractor, num_examples: int = 3000):
        self.extractor = extractor
        self.num_examples = num_examples
        self.examples: List[SkillsExample] = []

    def build_dataset(self) -> None:
        occupations = self.extractor.occupations
        for _ in range(self.num_examples):
            occ = random.choice(occupations)
            skills = self.extractor.get_skills_for_occupation(occ)
            if not skills:
                continue

            example = self._build_example_for_occupation(occ, skills)
            self.examples.append(example)

    def _build_example_for_occupation(
        self,
        occ: EscoOccupation,
        skills: List[EscoSkill]
    ) -> SkillsExample:
        # Job requirements: choose subset of skills
        job_skills = random.sample(skills, k=min(len(skills), random.randint(3, 8)))

        # CV skills: noisy subset (some missing, some extra)
        matched_skills = random.sample(job_skills, k=random.randint(1, len(job_skills)))
        extra_skills = random.sample(skills, k=min(2, len(skills)))
        cv_skills = list({s.name for s in (matched_skills + extra_skills)})

        job_requirements_json = [{"skill_name": s.name} for s in job_skills]
        matched_names = [s.name for s in matched_skills]
        missing_names = [s.name for s in job_skills if s.name not in matched_names]

        rationale = f"Matched {len(matched_names)} of {len(job_skills)} requirements; "
        if missing_names:
            rationale += f"missing: {', '.join(missing_names)}."
        else:
            rationale += "no missing skills."

        output_json = {
            "skills_evaluation": {
                "requirement_category": "Core Competencies & Skills",
                "job_requirements": job_requirements_json,
                "matched_cv_skills": matched_names,
                "missing_cv_skills": missing_names,
                "rationale": rationale
            }
        }

        instruction = self._build_instruction_text(occ, job_requirements_json, cv_skills)

        return SkillsExample(instruction=instruction, output=output_json)

    @staticmethod
    def _build_instruction_text(
        occ: EscoOccupation,
        job_requirements_json: List[Dict],
        cv_skills: List[str]
    ) -> str:
        job_skill_list = ", ".join([item["skill_name"] for item in job_requirements_json])
        cv_skill_list = ", ".join(cv_skills)

        return (
            f"You are evaluating a candidate for the role '{occ.title}'.\n"
            f"Job requirements (skills): {job_skill_list}.\n"
            f"Candidate CV skills: {cv_skill_list}.\n\n"
            f"Return a JSON object named 'skills_evaluation' that lists job_requirements, "
            f"matched_cv_skills, missing_cv_skills, and a short rationale."
        )


def run():
    extractor = EscoExtractor(
        skills_path=paths.raw_esco_skills,
        occupations_path=paths.raw_esco_occupations,
    )
    extractor.load()

    synthesizer = DatasetSynthesizer(extractor=extractor, num_examples=3000)
    synthesizer.build_dataset()

    with open(paths.processed_dataset, "w", encoding="utf-8") as f:
        for ex in synthesizer.examples:
            f.write(ex.to_jsonl() + "\n")

    print(f"Wrote {len(synthesizer.examples)} examples to {paths.processed_dataset}")



