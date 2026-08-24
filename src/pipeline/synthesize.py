# src/pipeline/synthesize.py

import json
import random
from typing import Dict, List, Optional, Tuple

from src.core.config import paths
from src.pipeline.extraction import EscoExtractor, EscoOccupation, EscoSkill

# ---------------------------------------------------------------------------
# System prompts — verbatim copies from the consumer project
# (CV_to_Job_Guestimator, src/prompts/templates.py). Training examples must
# match the served prompts byte-for-byte.
# ---------------------------------------------------------------------------

SKILL_MATCHER_SYSTEM_PROMPT = """Evaluate whether a candidate's CV satisfies each supplied job requirement.

Return exactly one evaluation for every numeric requirement_id supplied.
Set matched to true when the CV explicitly shows the skill_name or a directly equivalent skill_name.
Set matched to false when the CV does not show sufficient evidence.
Evaluate skill_name only. Do not require dated or commercial evidence here.
Do not extract, rename, summarize, or introduce skills in the evaluations.

Treat specific tools or certifications as evidence for a broader requirement only when they directly fulfill it."""

JOB_REQUIREMENTS_SYSTEM_PROMPT = """Extract atomic technical and operational skill_names from a job description.

Include specific domain tools, machinery, software, methodologies, frameworks, certifications, and technical skill_names.
Exclude generic soft skills such as hard working, communication, teamwork, and punctuality.
Return one skill_name per record. Split all combined requirements into separate records.
The skill_name field must contain only the skill name, without years-of-experience wording.
Return each unique skill_name once, grounded only in the job description.

Expected JSON structure:
{
  "job_requirements": [
    {
      "skill_name": ""
    },
    {
      "skill_name": ""
    }
  ]
}
"""

OVERALL_EXPERIENCE_SYSTEM_PROMPT = """Extract and classify the candidate's professional work experience against a target job.

1. Extract target job information from the job description:
- Extract target_job_title exactly as written.
- Extract target_overall_years from explicit minimum overall experience only.
- If the job description states a single minimum such as "3+ years", use that number.
- If a range is given such as "2-5 years", use the lower bound.
- If no minimum overall experience is stated, set target_overall_years to null.

2. Extract candidate roles from the CV:
- Extract ONLY paid employment or professional contractor work experience.
- STRICTLY EXCLUDE educational degrees, academic courses, certifications, bootcamps, personal projects, and volunteer work.
- Include role_title exactly as written.
- Format start_date and end_date as YYYY-MM or "Present" when present in the CV.
- Use null only if the CV is missing the start_date or end_date.
- Do not infer missing dates, missing employers, roles not explicitly stated, or experience not supported by the text.

3. Classify relevance for each role:
- Set is_relevant true only when the role provides directly transferable experience to the target job's responsibilities.
- Provide a brief, evidence-based match_rationale referencing specific job responsibilities and specific CV experience.
- Do not use generic statements such as "software experience is relevant".
- Do not assume skills or responsibilities not present in the text.

4. Output rules:
- Return valid JSON objects containing exactly one overall_experience object.
- Do not add fields not listed in the schema.
- Do not include markdown or trailing commentary.
- Do not hallucinate or infer missing data.

Expected JSON structure:
{
  "overall_experience": {
    "target_job_title": "",
    "target_overall_years": null,
    "candidate_roles": [
      {
        "role_title": "",
        "start_date": null,
        "end_date": null,
        "match_rationale": "",
        "is_relevant": false
      }
    ]
  }
}
"""


def compact_json(obj: Dict) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


class ChatExample:
    """One training example in OpenAI-style chat format, tagged with the
    agent task it trains so evaluate.py can score each type separately."""

    def __init__(self, task: str, system: str, user: str, assistant: str):
        self.task = task
        self.system = system
        self.user = user
        self.assistant = assistant

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "task": self.task,
                "messages": [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": self.user},
                    {"role": "assistant", "content": self.assistant},
                ],
            },
            ensure_ascii=False,
        )


class CvWriter:
    """Plans CV facts (roles, companies, dated periods, embedded skill
    mentions) and renders them as realistic prose. The facts are returned
    alongside the text so ground-truth labels are derived from the plan,
    never re-parsed from the rendering.

    The consumer pipeline redacts PII before its LLM calls, so CVs are
    rendered with the same [KIND] placeholders it produces.
    """

    COMPANIES = [
        "Harbourview Group", "Northgate Solutions", "Silverline Services",
        "Crestfield Ltd", "BlueRock Industries", "Kauri Point Trading",
        "Meridian Works", "Stonebridge Partners", "Pacific Rim Logistics",
        "Fernleaf Enterprises", "Ridgeline Contracting", "Bayside Operations",
        "Summit & Co", "Clearwater Holdings", "Ironbark Systems",
    ]

    SUMMARY_TEMPLATES = [
        "Reliable and adaptable professional with {years} years of experience as a {role}. "
        "Seeking to bring proven skills and a strong work ethic to a new team.",
        "Motivated {role} with {years} years of hands-on experience and a consistent "
        "record of delivering quality work on time.",
        "Experienced {role} with a practical, detail-focused approach developed over "
        "{years} years across a range of employers.",
        "Hardworking {role} offering {years} years of industry experience, known for "
        "dependability and a willingness to learn.",
    ]

    SKILL_BULLETS = [
        "Responsible for {skill} across multiple client projects.",
        "Applied {skill} on a daily basis to support team delivery targets.",
        "Gained extensive hands-on experience with {skill} in a fast-paced environment.",
        "Used {skill} to improve turnaround times and reduce errors.",
        "Trained and supported junior staff in {skill}.",
        "Delivered work that relied heavily on {skill}.",
        "Took ownership of tasks involving {skill} with minimal supervision.",
        "Completed projects requiring {skill} under tight deadlines.",
        "Recognised by management for strong results in {skill}.",
    ]

    FILLER_BULLETS = [
        "Collaborated with cross-functional teams to meet weekly targets.",
        "Prepared reports and documentation for management review.",
        "Liaised with clients and suppliers to resolve day-to-day issues.",
        "Supported health and safety compliance across the site.",
        "Mentored new team members during onboarding.",
        "Managed competing priorities in a busy environment.",
        "Maintained accurate records and followed company procedures.",
        "Contributed to continuous improvement initiatives within the team.",
    ]

    EDUCATION_FIELDS = [
        "Business Studies", "Applied Technology", "Operations Management",
        "Communication", "Project Management", "Workplace Leadership",
    ]

    INSTITUTIONS = [
        "Regional Institute of Technology", "City Polytechnic",
        "Open Learning Institute", "National Trades Academy",
    ]

    CERTIFICATION_LINES = [
        "First Aid Certificate ({year})",
        "Health and Safety Representative Certificate ({year})",
        "Forklift Operator Certification, National Trades Academy ({year})",
        "Workplace Fire Warden Training ({year})",
    ]

    VOLUNTEER_LINES = [
        "Volunteer Coordinator - Local Community Trust ({start_year} - {end_year})\n"
        "- Helped organise community events and fundraising activities.",
        "Volunteer - Regional Foodbank ({start_year} - {end_year})\n"
        "- Assisted with weekly sorting and distribution of donations.",
        "Volunteer Coach - Youth Sports Club ({start_year} - {end_year})\n"
        "- Ran weekend training sessions for junior teams.",
    ]

    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    @staticmethod
    def title_case(text: str) -> str:
        return " ".join(w if w.isupper() else w.capitalize() for w in text.split())

    def _ym(self, month_index: int, year: int) -> Tuple[str, str]:
        """Returns ("May 2021", "2021-05") for a month index and year."""
        return f"{self.MONTHS[month_index]} {year}", f"{year}-{month_index + 1:02d}"

    def plan_roles(
        self,
        titles_with_mentions: List[Tuple[str, List[str]]],
        omit_dates_prob: float = 0.0,
        present_prob: float = 0.3,
    ) -> List[Dict]:
        """Builds role facts (most recent first): rendered title, company,
        period text as it will appear in the heading, and the ground-truth
        YYYY-MM dates derived from the same plan."""
        facts: List[Dict] = []
        companies = random.sample(self.COMPANIES, k=len(titles_with_mentions))
        end_year = random.choice([2024, 2025, 2026])
        for i, (title, mentions) in enumerate(titles_with_mentions):
            is_present = i == 0 and random.random() < present_prob
            end_month = random.randrange(12)
            start_year = end_year - random.randint(1, 4)
            start_month = random.randrange(12)
            start_text, start_ym = self._ym(start_month, start_year)

            if is_present:
                period_text, end_value = f"{start_text} - Present", "Present"
            else:
                end_text, end_ym = self._ym(end_month, end_year)
                period_text, end_value = f"{start_text} - {end_text}", end_ym

            if random.random() < omit_dates_prob:
                period_text, start_ym, end_value = None, None, None

            facts.append({
                "title": self.title_case(title),
                "company": companies[i],
                "period_text": period_text,
                "start_date": start_ym,
                "end_date": end_value,
                "mentions": list(mentions),
            })
            end_year = start_year - random.randint(0, 1)
        return facts

    def render_cv(
        self,
        role_facts: List[Dict],
        key_skills: Optional[List[str]] = None,
        include_certifications: bool = False,
        include_volunteer: bool = False,
    ) -> str:
        lines: List[str] = []

        # Every CV reaching the consumer's skill_match / experience agents
        # has already been through PII redaction, so every synthetic CV is
        # rendered post-redaction: PII spans appear as [KIND] tokens, while
        # employers, job titles, institutions, skills, and employment dates
        # are never redacted and stay plain text. No raw PII is ever emitted.
        lines.append("[PERSON_NAME]")
        lines.append(" | ".join(["[OTHER_IDENTIFIER]"] * random.randint(1, 3)))
        if random.random() < 0.06:
            lines.append("[STREET_ADDRESS]")
        lines.append("")

        if random.random() < 0.08:
            details = [
                d for d in [
                    "Date of birth: [DATE_OF_BIRTH]",
                    "Nationality: [NATIONALITY]",
                    "Marital status: [MARITAL_OR_FAMILY]",
                ] if random.random() < 0.6
            ]
            if details:
                lines.append("PERSONAL DETAILS")
                lines.extend(details)
                lines.append("")

        years = random.randint(3, 20)
        lines.append("PROFESSIONAL SUMMARY")
        lines.append(random.choice(self.SUMMARY_TEMPLATES).format(
            years=years, role=role_facts[0]["title"]))
        lines.append("")

        lines.append("WORK EXPERIENCE")
        lines.append("")
        for fact in role_facts:
            heading = f"{fact['title']} - {fact['company']}"
            if fact["period_text"]:
                heading += f" ({fact['period_text']})"
            lines.append(heading)

            bullets = [
                random.choice(self.SKILL_BULLETS).format(skill=mention)
                for mention in fact["mentions"]
            ]
            for _ in range(random.randint(1, 2)):
                bullets.append(random.choice(self.FILLER_BULLETS))
            random.shuffle(bullets)
            lines.extend(f"- {bullet}" for bullet in bullets)
            lines.append("")

        if key_skills:
            lines.append("KEY SKILLS")
            lines.extend(f"- {skill}" for skill in key_skills)
            lines.append("")

        lines.append("EDUCATION")
        lines.append(
            f"{random.choice(['Diploma', 'Certificate', 'National Certificate'])} in "
            f"{random.choice(self.EDUCATION_FIELDS)}, {random.choice(self.INSTITUTIONS)} "
            f"({random.randint(2005, 2020)})"
        )
        lines.append("")

        if include_certifications:
            lines.append("CERTIFICATIONS")
            for template in random.sample(self.CERTIFICATION_LINES, k=random.randint(1, 2)):
                lines.append(f"- {template.format(year=random.randint(2015, 2025))}")
            lines.append("")

        if include_volunteer:
            start_year = random.randint(2014, 2021)
            lines.append("VOLUNTEER WORK")
            lines.append(random.choice(self.VOLUNTEER_LINES).format(
                start_year=start_year, end_year=start_year + random.randint(1, 3)))
            lines.append("")

        lines.append("REFERENCES")
        if random.random() < 0.08:
            for _ in range(random.randint(1, 2)):
                lines.append("[REFEREE] - [OTHER_IDENTIFIER]")
        else:
            lines.append("Available upon request.")

        return "\n".join(lines)


class JobAdWriter:
    """Renders job-listing text (full ads and requirements sections) from a
    planned set of skills, returning the ground-truth facts alongside."""

    SECTION_HEADERS = ["Requirements:", "What you'll need:", "About you:", "Skills and experience:"]

    PLAIN_TEMPLATES = [
        "Proficiency in {a}.",
        "Demonstrated experience with {a}.",
        "Strong background in {a}.",
        "Competence in {a} is essential.",
        "Proven ability in {a}.",
    ]

    PAIR_TEMPLATES = [
        "Experience with {a} and {b}.",
        "Proficiency in {a} as well as {b}.",
        "Strong skills in both {a} and {b}.",
    ]

    YEARS_TEMPLATES = [
        "{n}+ years' experience with {a}.",
        "At least {n} years working with {a}.",
        "Minimum {n} years' experience in {a}.",
    ]

    SOFT_SKILL_BULLETS = [
        "Excellent communication skills.",
        "A great team player with a hard working attitude.",
        "Reliable, punctual and well presented.",
        "Strong interpersonal and teamwork skills.",
        "A positive attitude and willingness to learn.",
    ]

    OVERALL_YEARS_SINGLE = [
        "A minimum of {n}+ years' overall experience is required.",
        "At least {n} years of professional experience is expected.",
        "Applicants should have {n}+ years' experience in a similar position.",
    ]

    OVERALL_YEARS_RANGE = [
        "{a}-{b} years of overall professional experience is required.",
        "We are looking for someone with {a}-{b} years' experience.",
    ]

    def requirements_section(self, skills: List[str]) -> Tuple[str, List[str]]:
        """Renders a requirements/qualifications section. Returns the text
        and the ordered ground-truth skill names planted in it."""
        bullets: List[str] = []
        gt_names: List[str] = []

        queue = list(skills)
        while queue:
            skill = queue.pop(0)
            roll = random.random()
            if roll < 0.3 and queue:
                other = queue.pop(0)
                bullets.append(random.choice(self.PAIR_TEMPLATES).format(a=skill, b=other))
                gt_names.extend([skill, other])
            elif roll < 0.55:
                bullets.append(random.choice(self.YEARS_TEMPLATES).format(
                    a=skill, n=random.randint(1, 8)))
                gt_names.append(skill)
            else:
                bullets.append(random.choice(self.PLAIN_TEMPLATES).format(a=skill))
                gt_names.append(skill)

        for soft in random.sample(self.SOFT_SKILL_BULLETS, k=random.randint(1, 2)):
            bullets.insert(random.randint(0, len(bullets)), soft)

        text = random.choice(self.SECTION_HEADERS) + "\n" + "\n".join(f"- {b}" for b in bullets)
        return text, gt_names

    def full_listing(
        self, job_title: str, company: str, responsibilities: List[str]
    ) -> Tuple[str, Optional[float]]:
        """Renders a full job ad (title line included). Returns the text and
        the planted minimum overall-experience years (None when omitted)."""
        lines = [job_title, ""]
        lines.append(
            f"{company} is looking for a dedicated {job_title} to join our team on a "
            f"{random.choice(['full-time', 'permanent', 'fixed-term'])} basis."
        )
        lines.append("")
        lines.append("Key responsibilities:")
        lines.extend(f"- {r}" for r in responsibilities)
        lines.append("")

        years_value: Optional[float] = None
        roll = random.random()
        if roll < 1 / 3:
            pass  # no overall-experience statement -> null
        elif roll < 2 / 3:
            n = random.randint(1, 8)
            years_value = float(n)
            lines.append(random.choice(self.OVERALL_YEARS_SINGLE).format(n=n))
            lines.append("")
        else:
            a = random.randint(1, 5)
            b = a + random.randint(1, 4)
            years_value = float(a)
            lines.append(random.choice(self.OVERALL_YEARS_RANGE).format(a=a, b=b))
            lines.append("")

        lines.append("Competitive remuneration and a supportive team environment on offer.")
        return "\n".join(lines), years_value


class DatasetSynthesizer:
    """Builds one mixed dataset of chat-format training examples for the
    three agent roles the fine-tuned model serves in CV_to_Job_Guestimator:
    ~50% skill_match, ~25% requirements, ~25% experience."""

    def __init__(self, extractor: EscoExtractor, num_examples: int = 3000):
        self.extractor = extractor
        self.num_examples = num_examples
        self.examples: List[ChatExample] = []
        self.cv_writer = CvWriter()
        self.ad_writer = JobAdWriter()

    def build_dataset(self) -> None:
        occupations = [o for o in self.extractor.occupations if o.skill_ids]
        for _ in range(self.num_examples):
            occ = random.choice(occupations)
            skills = self._unique_by_name(self.extractor.get_skills_for_occupation(occ))
            if len(skills) < 3:
                continue

            roll = random.random()
            if roll < 0.5:
                example = self._build_skill_match_example(occ, skills, occupations)
            elif roll < 0.75:
                example = self._build_requirements_example(skills)
            else:
                example = self._build_experience_example(occ, skills, occupations)

            if example is not None:
                self.examples.append(example)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_by_name(skills: List[EscoSkill]) -> List[EscoSkill]:
        seen: Dict[str, EscoSkill] = {}
        for skill in skills:
            seen.setdefault(skill.name.strip().casefold(), skill)
        return list(seen.values())

    @staticmethod
    def _surface_form(skill: EscoSkill) -> str:
        """How the skill appears in the CV: usually verbatim, sometimes a
        directly equivalent alternative label (paraphrase)."""
        if skill.alt_labels and random.random() < 0.4:
            return random.choice(skill.alt_labels)
        return skill.name

    def _other_titles(self, occupations: List[EscoOccupation], exclude_id: str, k: int) -> List[str]:
        return [o.title for o in random.sample(occupations, k=min(len(occupations), k + 1))
                if o.occ_id != exclude_id][:k]

    # ------------------------------------------------------------------
    # Type 1: skill matcher
    # ------------------------------------------------------------------

    def _build_skill_match_example(
        self,
        occ: EscoOccupation,
        skills: List[EscoSkill],
        occupations: List[EscoOccupation],
    ) -> Optional[ChatExample]:
        job_skills = random.sample(skills, k=min(len(skills), random.randint(3, 8)))

        # Matched subset with a deliberate label mix: ~8% all-false, ~20%
        # all-true, and the rest mixed so most examples carry both matched
        # and unmatched requirements.
        roll = random.random()
        if roll < 0.08:
            matched_skills: List[EscoSkill] = []
        elif roll < 0.28 or len(job_skills) == 1:
            matched_skills = list(job_skills)
        else:
            matched_skills = random.sample(
                job_skills, k=random.randint(1, len(job_skills) - 1)
            )
        matched_keys = {s.name.casefold() for s in matched_skills}
        unmatched_names = [s.name.casefold() for s in job_skills if s.name.casefold() not in matched_keys]

        # Distractor skills: present in the CV but not part of the job
        # requirements, so the model learns not to be fooled by extras.
        requirement_keys = {s.name.casefold() for s in job_skills}
        all_skills = list(self.extractor.skills.values())
        distractors: List[EscoSkill] = []
        for candidate in random.sample(all_skills, k=min(len(all_skills), 12)):
            if len(distractors) >= random.randint(1, 3):
                break
            key = candidate.name.casefold()
            if key in requirement_keys:
                continue
            # A distractor must not accidentally provide evidence for an
            # unmatched requirement.
            if any(u in key or key in u for u in unmatched_names):
                continue
            distractors.append(candidate)

        # Near-miss distractors: skills from the SAME occupation that were
        # not required. They make the CV read as domain-plausible for the
        # unmatched requirements, which the model must still label false.
        near_miss_pool = [
            s for s in skills
            if s.name.casefold() not in requirement_keys
            and not any(u in s.name.casefold() or s.name.casefold() in u for u in unmatched_names)
        ]
        near_miss = random.sample(
            near_miss_pool, k=min(len(near_miss_pool), random.randint(0, 2))
        )

        mentions = [self._surface_form(s) for s in matched_skills]
        mentions += [self._surface_form(s) for s in distractors + near_miss]
        random.shuffle(mentions)

        titles = [occ.title] + self._other_titles(occupations, occ.occ_id, 2)
        titles = titles[: random.randint(2, 3)] if mentions else titles[:2]
        titles_with_mentions: List[Tuple[str, List[str]]] = [(t, []) for t in titles]
        for idx, mention in enumerate(mentions):
            titles_with_mentions[idx % len(titles)][1].append(mention)

        role_facts = self.cv_writer.plan_roles(titles_with_mentions)
        key_skills = (
            random.sample(mentions, k=min(len(mentions), random.randint(2, 4)))
            if mentions and random.random() < 0.3
            else None
        )
        cv_text = self.cv_writer.render_cv(role_facts, key_skills=key_skills)

        # Consistency pass: if an "unmatched" requirement name still ended up
        # verbatim in the CV text, the correct label for it is matched.
        cv_lower = cv_text.casefold()
        evaluations = []
        for requirement_id, skill in enumerate(job_skills):
            key = skill.name.casefold()
            matched = key in matched_keys or key in cv_lower
            evaluations.append({"requirement_id": requirement_id, "matched": matched})

        # User message: byte-for-byte the format built by the consumer's
        # SkillMatcherAgent (json.dumps with default separators).
        requirements_payload = json.dumps(
            [
                {"requirement_id": requirement_id, "skill_name": skill.name}
                for requirement_id, skill in enumerate(job_skills)
            ],
            ensure_ascii=False,
        )
        user_message = f"JOB REQUIREMENTS:\n{requirements_payload}\n\nCANDIDATE CV:\n{cv_text}"

        return ChatExample(
            task="skill_match",
            system=SKILL_MATCHER_SYSTEM_PROMPT,
            user=user_message,
            assistant=compact_json({"evaluations": evaluations}),
        )

    # ------------------------------------------------------------------
    # Type 2: job requirements extraction
    # ------------------------------------------------------------------

    def _build_requirements_example(self, skills: List[EscoSkill]) -> Optional[ChatExample]:
        job_skills = random.sample(skills, k=min(len(skills), random.randint(4, 8)))
        section_text, gt_names = self.ad_writer.requirements_section(
            [s.name for s in job_skills]
        )

        # User message: byte-for-byte the format built by the consumer's
        # JobRequirementsAgent (the requirements section, not the whole ad).
        user_message = f"JOB DESCRIPTION:\n{section_text}"

        return ChatExample(
            task="requirements",
            system=JOB_REQUIREMENTS_SYSTEM_PROMPT,
            user=user_message,
            assistant=compact_json(
                {"job_requirements": [{"skill_name": name} for name in gt_names]}
            ),
        )

    # ------------------------------------------------------------------
    # Type 3: overall work experience
    # ------------------------------------------------------------------

    RELEVANT_TITLE_VARIANTS = ["{t}", "{t}", "Senior {t}", "Junior {t}"]

    def _build_experience_example(
        self,
        occ: EscoOccupation,
        skills: List[EscoSkill],
        occupations: List[EscoOccupation],
    ) -> Optional[ChatExample]:
        job_title = CvWriter.title_case(occ.title)
        responsibilities = [s.name for s in random.sample(skills, k=min(len(skills), random.randint(3, 5)))]
        listing_text, years_value = self.ad_writer.full_listing(
            job_title, random.choice(CvWriter.COMPANIES), responsibilities
        )

        # Plan candidate roles: a mix of relevant (target occupation, with
        # bullets that reuse listing responsibilities) and irrelevant ones.
        num_roles = random.randint(1, 3)
        role_specs: List[Tuple[str, List[str]]] = []
        role_meta: List[Dict] = []
        for _ in range(num_roles):
            if random.random() < 0.6:
                title = random.choice(self.RELEVANT_TITLE_VARIANTS).format(t=occ.title)
                overlap = random.sample(responsibilities, k=min(len(responsibilities), random.randint(1, 2)))
                role_specs.append((title, list(overlap)))
                role_meta.append({"is_relevant": True, "overlap": overlap})
            else:
                other = random.choice([o for o in occupations if o.occ_id != occ.occ_id])
                other_skills = self._unique_by_name(self.extractor.get_skills_for_occupation(other))
                mentions = [s.name for s in random.sample(other_skills, k=min(len(other_skills), random.randint(0, 2)))]
                role_specs.append((other.title, mentions))
                role_meta.append({"is_relevant": False, "mentions": mentions})

        role_facts = self.cv_writer.plan_roles(role_specs, omit_dates_prob=0.15)
        cv_text = self.cv_writer.render_cv(
            role_facts,
            include_certifications=random.random() < 0.5,
            include_volunteer=random.random() < 0.5,
        )

        candidate_roles = []
        for fact, meta in zip(role_facts, role_meta):
            if meta["is_relevant"]:
                skill = random.choice(meta["overlap"])
                rationale = random.choice([
                    f"Hands-on work with {skill} directly covers a key responsibility of the {job_title} role.",
                    f"The role involved {skill}, which the {job_title} listing names as a core responsibility.",
                    f"Experience applying {skill} transfers directly to the listing's responsibility for {skill}.",
                ])
            else:
                resp = random.choice(responsibilities)
                focus = meta["mentions"][0] if meta["mentions"] else f"{fact['title'].lower()} duties"
                rationale = random.choice([
                    f"The role centred on {focus} and shows no overlap with responsibilities such as {resp}.",
                    f"Work focused on {focus}, which does not transfer to the {job_title} responsibility for {resp}.",
                ])
            candidate_roles.append({
                "role_title": fact["title"],
                "start_date": fact["start_date"],
                "end_date": fact["end_date"],
                "match_rationale": rationale,
                "is_relevant": meta["is_relevant"],
            })

        # User message: byte-for-byte the format built by the consumer's
        # OverallExperienceAgent (full listing text, then the CV).
        user_message = f"JOB DESCRIPTION:\n{listing_text}\n\nCANDIDATE CV:\n{cv_text}"

        return ChatExample(
            task="experience",
            system=OVERALL_EXPERIENCE_SYSTEM_PROMPT,
            user=user_message,
            assistant=compact_json({
                "overall_experience": {
                    "target_job_title": job_title,
                    "target_overall_years": years_value,
                    "candidate_roles": candidate_roles,
                }
            }),
        )


def run():
    extractor = EscoExtractor(
        skills_path=paths.raw_esco_skills,
        occupations_path=paths.raw_esco_occupations,
    )
    extractor.load()

    synthesizer = DatasetSynthesizer(extractor=extractor, num_examples=3000)
    synthesizer.build_dataset()

    counts: Dict[str, int] = {}
    with open(paths.processed_dataset, "w", encoding="utf-8") as f:
        for ex in synthesizer.examples:
            counts[ex.task] = counts.get(ex.task, 0) + 1
            f.write(ex.to_jsonl() + "\n")

    print(f"Wrote {len(synthesizer.examples)} examples to {paths.processed_dataset}")
    for task, count in sorted(counts.items()):
        print(f"  {task}: {count}")
