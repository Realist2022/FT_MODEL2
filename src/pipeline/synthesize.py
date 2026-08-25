# src/pipeline/synthesize.py

import json
import os
import random
import time
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from tqdm import tqdm

from src.core.config import paths, synthesis
from src.core.schema import JobRequirementsOutput, OverallExperienceResponse, SkillEvaluationDecision
from src.pipeline.extraction import EscoExtractor, EscoOccupation, EscoSkill

# ---------------------------------------------------------------------------
# System prompts — verbatim copies from the consumer project
# (CV_to_Job_Guestimator, src/prompts/templates.py). Training examples must
# match the served prompts byte-for-byte.
# ---------------------------------------------------------------------------

JOB_REQUIREMENTS_SYSTEM_PROMPT = """Extract atomic technical and operational skill_names from a job description.

Include specific domain tools, machinery, software, methodologies, frameworks, certifications, and technical skill_names.
Exclude generic soft skills such as hard working, communication, teamwork, and punctuality.
Exclude vague qualitative descriptors that name no specific technology, tool, or named
methodology, such as "modern development practices", "modern engineering practices",
"modern web applications", or "fast-paced environment" — these are not independently
verifiable against a CV. Keep concrete, named methodologies and practices such as
"Agile", "CI/CD", "code reviews", or "test-driven development".
Return one skill_name per record. Split all combined requirements into separate records.
The skill_name field must contain only the skill name, without years-of-experience wording.
Return each unique skill_name once, grounded only in the job description.

When a requirement names a general category followed by a specific example marked as
optional or preferred (in parentheses, or after "e.g."/"such as"/"including"), extract
only the general category as the skill_name and drop the example and its qualifier.
Example: "Exposure to cloud technologies (AWS preferred)" -> skill_name: "Cloud technologies".
Do not create a second record for the example in this case.
Only extract the specific named technology on its own when the text requires that
technology directly, not merely as an example of a broader category.

When a requirement names a general category and then states the exact required
instantiation of that category in the same sentence (e.g. via "specifically
requiring", "specifically", or by naming the only acceptable tool/method), treat
the general category and the specific instantiation as ONE requirement, not two.
Extract only the specific instantiation as the skill_name, and do not create a
separate record for the general category that wraps it.
Example: "A strong foundation in digital literacy, specifically requiring 2 years
of experience using Google Workspace for Education" -> skill_name: "Google
Workspace for Education". Do not also create a record for "digital literacy".

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

SKILL_MATCHER_SYSTEM_PROMPT = """Evaluate whether a candidate's CV satisfies each supplied job requirement.

Return exactly one evaluation for every numeric requirement_id supplied.
Set matched to true when the CV explicitly shows the skill_name or a directly equivalent skill_name.
Set matched to false when the CV does not show sufficient evidence.
Evaluate skill_name only. Do not require dated or commercial evidence here.
Do not extract, rename, summarize, or introduce skills in the evaluations.

Treat specific tools or certifications as evidence for a broader requirement only when they directly fulfill it.

Apply category-inclusion reasoning: when a requirement names a general category or
practice, a specific CV item that is a well-known member of that category counts as
a match, even if the CV never uses the requirement's exact wording.
- "Relational databases" is satisfied by any named relational database the CV lists
  (e.g. MySQL, PostgreSQL, SQLite, Oracle, SQL Server), not only PostgreSQL itself.
- "Cloud technologies" is satisfied by any named cloud provider or service the CV
  lists (e.g. AWS, Azure, GCP), even under a different section heading such as
  "DevOps & Cloud".
- A practice-based requirement such as "AI-assisted software development" is
  satisfied by concrete CV evidence of that practice — a project description,
  tool, or self-description naming AI/LLM/chatbot work — not only by the literal
  phrase appearing in the CV.
- A professional registration or licensing requirement (e.g. "New Zealand
  Practising Certificate") is satisfied by CV evidence of full registration or
  licensure with the relevant regulatory body, even when worded differently
  (e.g. "Fully Registered Teacher (NZTC)").
- A jurisdiction- or system-specific experience requirement (e.g. "teaching
  experience in a New Zealand secondary school") is satisfied by CV evidence
  that uses that jurisdiction's characteristic terminology, curriculum, or role
  titles (e.g. NCEA levels, a Wellington-based school, Dean/Form Teacher roles),
  even without the literal phrase appearing.

These are illustrative examples from a few domains, not an exhaustive list — apply
the same category-inclusion and equivalent-terminology reasoning in whatever
professional domain the CV and requirement belong to (medicine, trades, education,
finance, etc.), not only software or technology.

Ground every match in text that actually appears in the CV. Do not infer a category
match from a requirement alone, and do not invent CV content that is not present."""

OVERALL_EXPERIENCE_SYSTEM_PROMPT = """Extract and classify the candidate's professional work experience against a target job.

1. Extract target job information from the job description:
- Extract target_job_title exactly as written.
- Extract target_overall_years from explicit minimum overall experience only.
- If the job description states a single minimum such as "3+ years", use that number.
- If a range is given such as "2-5 years", use the lower bound.
- Treat a years figure as the overall requirement even when it is phrased alongside
  the role's core/primary technologies (e.g. "2-5 years' commercial experience with
  React and Node.js" -> target_overall_years: 2), since that is the role's experience bar.
- Only leave target_overall_years null when years are tied to a narrow, secondary
  tool/certification unrelated to the role's main responsibilities, or no years figure
  appears anywhere in the listing.

2. Extract candidate roles from the CV:
- Extract ONLY paid employment or professional contractor work experience.
- STRICTLY EXCLUDE educational degrees, academic courses, certifications, bootcamps, personal projects, and volunteer work.
- Include role_title exactly as written.
- Format start_date and end_date as YYYY-MM or "Present" when present in the CV.
- Use null only if the CV is missing the start_date or end_date.
- Do not infer missing dates, missing employers, roles not explicitly stated, or experience not supported by the text.

3. Classify relevance for each role:
- Set is_relevant true only when the role provides directly transferable experience to the target job's responsibilities.
- Base is_relevant on the role's full set of listed duties/bullets, not just one representative bullet — a role is relevant if ANY of its responsibilities directly matches a target job responsibility, even if other bullets in the same role do not.
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


# ---------------------------------------------------------------------------
# ESCO skill names are conventionally bare-infinitive verb phrases ("manage
# artistic career", "develop strategy to solve problems") -- but CvWriter's
# SKILL_BULLETS templates all supply their own leading verb/preposition
# ("Applied {skill}...", "Trained ... in {skill}."), so a literal skill name
# collides with it the same way an ungoverned LLM paraphrase did (see
# SkillParaphraser._PROMPT's comment). This table rewrites the leading verb
# to its gerund form ("managing artistic career") so it reads as a noun
# phrase in every template slot instead. It only fires on skills that are
# NOT going through SkillParaphraser (which already produces gerund/noun
# phrases directly) -- see _surface_form.
#
# Deliberately conservative, not a general English conjugator: this lookup
# covers the ~100 most frequent leading verbs across the actual ESCO
# taxonomy (data/raw/esco_skills.json), measured directly rather than
# guessed -- together they cover roughly 60% of all skill-name leading
# words. Anything not in the table (including genuine noun-phrase skill
# names like "PostgreSQL" or "customer service") is left completely
# unchanged rather than guessed at.
# ---------------------------------------------------------------------------

_LEADING_VERB_GERUNDS: Dict[str, str] = {
    "manage": "managing", "operate": "operating", "perform": "performing",
    "maintain": "maintaining", "use": "using", "develop": "developing",
    "monitor": "monitoring", "provide": "providing", "apply": "applying",
    "prepare": "preparing", "advise": "advising", "ensure": "ensuring",
    "design": "designing", "create": "creating", "understand": "understanding",
    "tend": "tending", "conduct": "conducting", "analyse": "analysing",
    "assess": "assessing", "install": "installing", "write": "writing",
    "handle": "handling", "inspect": "inspecting", "identify": "identifying",
    "supervise": "supervising", "work": "working", "assist": "assisting",
    "plan": "planning", "teach": "teaching", "carry": "carrying",
    "coordinate": "coordinating", "set": "setting", "interact": "interacting",
    "test": "testing", "evaluate": "evaluating", "check": "checking",
    "promote": "promoting", "organise": "organising", "follow": "following",
    "repair": "repairing", "implement": "implementing", "clean": "cleaning",
    "select": "selecting", "communicate": "communicating", "sell": "selling",
    "keep": "keeping", "interpret": "interpreting", "assemble": "assembling",
    "adjust": "adjusting", "control": "controlling", "support": "supporting",
    "make": "making", "liaise": "liaising", "determine": "determining",
    "collect": "collecting", "oversee": "overseeing", "report": "reporting",
    "train": "training", "define": "defining", "remove": "removing",
    "calculate": "calculating", "measure": "measuring", "negotiate": "negotiating",
    "process": "processing", "adapt": "adapting", "produce": "producing",
    "examine": "examining", "prevent": "preventing", "study": "studying",
    "build": "building", "cut": "cutting", "comply": "complying",
    "read": "reading", "research": "researching", "demonstrate": "demonstrating",
    "administer": "administering", "transfer": "transferring", "execute": "executing",
    "record": "recording", "lead": "leading", "draw": "drawing",
    "drive": "driving", "diagnose": "diagnosing", "attend": "attending",
    "participate": "participating", "estimate": "estimating", "instruct": "instructing",
    "establish": "establishing", "store": "storing", "treat": "treating",
    "arrange": "arranging", "review": "reviewing", "take": "taking",
    "integrate": "integrating",
}


def _gerundize_if_verb_led(phrase: str) -> str:
    first, _, rest = phrase.partition(" ")
    gerund = _LEADING_VERB_GERUNDS.get(first.casefold())
    if gerund is None or not rest:
        return phrase
    return f"{gerund} {rest}"


# ---------------------------------------------------------------------------
# Instructor's Mode.JSON schema suffix — reproduced byte-for-byte from
# instructor/v2/providers/openai/handlers.py (both the OpenAI and streaming
# handlers build this identically): a system-prompt suffix Instructor
# appends to every real request, containing the response model's full JSON
# schema. This is NOT optional flavour text -- CV_to_Job_Guestimator's
# diagnosis of cv-guestimator's rigid literal-matching found that this exact
# block, sitting between the developer-authored reasoning instructions and
# the CV/JD content, measurably changes the model's decisions. A model
# trained only on the bare system prompt is being trained on a different
# input distribution than the one it's served at inference time -- this
# closes that gap the same way build_training_dataset.py does on the
# consumer side (by capturing/reproducing what Instructor actually sends,
# rather than hand-approximating it).
#
# Only the top-level "title" varies from the response_model's own class
# name: the skill-matcher agent serves a request-scoped subclass of
# SkillEvaluationDecision built via create_model("ConstrainedSkillEvaluation
# Decision", __base__=SkillEvaluationDecision, ...) in the consumer's
# src/services/agents.py -- the extra validator it adds doesn't change the
# JSON schema's fields, only its title, which is reproduced with
# title_override below.
# ---------------------------------------------------------------------------


def _instructor_json_suffix(model: Type[BaseModel], title_override: Optional[str] = None) -> str:
    schema = model.model_json_schema()
    if title_override:
        schema["title"] = title_override
    return dedent(
        f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(schema, indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
    )


JOB_REQUIREMENTS_SYSTEM_PROMPT += "\n\n" + _instructor_json_suffix(JobRequirementsOutput)
SKILL_MATCHER_SYSTEM_PROMPT += "\n\n" + _instructor_json_suffix(
    SkillEvaluationDecision, title_override="ConstrainedSkillEvaluationDecision"
)
OVERALL_EXPERIENCE_SYSTEM_PROMPT += "\n\n" + _instructor_json_suffix(OverallExperienceResponse)


class _ParaphraseResponse(BaseModel):
    phrase: str


class SkillParaphraser:
    """Generates and caches a genuinely differently-worded, real-world
    equivalent CV phrase for an ESCO skill, via OpenAI.

    Why this exists: without it, every "matched: true" label in the dataset
    is recoverable by a literal (or ESCO-alt-label) substring check against
    the CV text (see the consistency-pass comment in
    DatasetSynthesizer._build_skill_match_example) -- the model is never
    shown a case where genuinely different wording still satisfies a
    requirement, so it has no training signal to learn that reasoning from,
    and the "near-miss distractor" mechanism actively rewards it for
    treating anything non-literal as false. This closes that gap for a
    controlled fraction of matched skills (see SynthesisConfig.llm_paraphrase_prob).

    Each rewrite is verified to not just restate the skill name/alt_label
    before being accepted, and results are cached to disk by skill_id so
    repeated dataset-build runs don't re-spend API calls on the same skill --
    ESCO skills repeat heavily across the ~3000 sampled occupations.
    """

    MODEL = "gpt-4o"

    # This phrase gets substituted into an *existing* CV bullet template that
    # already supplies its own leading verb/preposition (e.g. "Used {phrase}
    # to improve turnaround times", "Recognised ... for strong results in
    # {phrase}") -- see CvWriter.SKILL_BULLETS. So the phrase itself MUST be
    # a bare noun/gerund phrase, never a clause with its own verb, or the
    # composed sentence reads as broken ("Used Oversee legal data
    # compilation...", "results in Expertise in touch typing..."). The
    # explicit shape instruction + good/bad examples below exist specifically
    # to prevent that -- an earlier version of this prompt only constrained
    # length, and GPT-4o would often lead with its own verb.
    _PROMPT = (
        "A job requirement is: \"{skill}\".\n"
        "Write ONE short CV-style NOUN/GERUND PHRASE (roughly 3-10 words, no "
        "trailing period) describing how a real person's experience would "
        "satisfy this requirement, WITHOUT using the words \"{skill}\" or "
        "any of: {alt_labels}.\n"
        "It must grammatically complete a sentence like \"Responsible for "
        "___\" or \"Trained others in ___\" -- so it must NOT be a full "
        "clause and must NOT start with a verb (e.g. \"Managed\", "
        "\"Oversaw\", \"Ensured\", \"Directed\", \"Led\", \"Tracked\") or a "
        "phrase like \"Expertise in\" / \"Proficient in\".\n"
        "Good: \"touch typing at 70+ words per minute\", \"legal document "
        "compilation using Relativity\", \"OSHA-compliant safety "
        "inspections\".\n"
        "Bad (starts with a verb, would read as a broken double-verb once "
        "slotted into a template): \"Managed touch typing\", \"Oversee "
        "legal document compilation\", \"Ensured OSHA-compliant safety\".\n"
        "Lowercase the first word unless it is a proper noun. Name it the "
        "way it would actually appear on a real CV -- a specific "
        "certification, system, tool, or concrete duty -- not a generic "
        "rewording of the requirement's own phrasing."
    )

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache: Dict[str, Optional[str]] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()  # picks up OPENAI_API_KEY from the environment
        return self._client

    def get(self, skill: EscoSkill) -> Optional[str]:
        """Returns a cached/generated paraphrase, or None if generation
        failed or was rejected -- callers should fall back to the literal
        name/alt_label in that case."""
        if skill.skill_id in self.cache:
            return self.cache[skill.skill_id]
        phrase = self._generate(skill)
        self.cache[skill.skill_id] = phrase
        return phrase

    def _generate(self, skill: EscoSkill, max_retries: int = 3) -> Optional[str]:
        import openai

        client = self._client_or_create()
        prompt = self._PROMPT.format(
            skill=skill.name,
            alt_labels=", ".join(skill.alt_labels) or "none",
        )
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.parse(
                    model=self.MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=_ParaphraseResponse,
                )
                message = response.choices[0].message
                if message.parsed is None:
                    # The model refused instead of answering, or the SDK
                    # couldn't parse a result -- message.refusal carries the
                    # reason when it's a refusal.
                    raise ValueError(message.refusal or "no parsed response returned")
                phrase = message.parsed.phrase.strip()
                break
            except openai.RateLimitError:
                wait = 5 * (attempt + 1)
                print(f"  [paraphrase] rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            except Exception as exc:
                print(f"  [paraphrase] {skill.name!r}: generation failed ({exc}); using literal wording")
                return None
        else:
            print(f"  [paraphrase] {skill.name!r}: gave up after {max_retries} rate-limit retries; using literal wording")
            return None

        banned = {skill.name.casefold(), *(a.casefold() for a in skill.alt_labels)}
        if not phrase or any(term in phrase.casefold() for term in banned):
            print(f"  [paraphrase] {skill.name!r}: rewrite still contained the literal term; using literal wording")
            return None
        return phrase

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )


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

    # None of these may start with a word sharing a lemma with any value in
    # _LEADING_VERB_GERUNDS ("Applied"/apply, "Used"/use, "Trained"/train
    # were the three that did -- e.g. a gerundized "use" skill plugged into
    # the old "Used {skill}..." template produced "Used using X...", the
    # same double-verb collision _gerundize_if_verb_led exists to prevent).
    SKILL_BULLETS = [
        "Responsible for {skill} across multiple client projects.",
        "Contributed to {skill} on a daily basis to support team delivery targets.",
        "Gained extensive hands-on experience with {skill} in a fast-paced environment.",
        "Leveraged {skill} to improve turnaround times and reduce errors.",
        "Mentored junior staff in {skill}.",
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

    def __init__(
        self,
        extractor: EscoExtractor,
        num_examples: int = 3000,
        paraphraser: Optional[SkillParaphraser] = None,
    ):
        self.extractor = extractor
        self.num_examples = num_examples
        self.examples: List[ChatExample] = []
        self.cv_writer = CvWriter()
        self.ad_writer = JobAdWriter()
        # None disables the LLM-paraphrase surface form entirely (falls back
        # to literal name / ESCO alt_label) -- see run()'s OPENAI_API_KEY guard.
        self.paraphraser = paraphraser

    def build_dataset(self) -> None:
        occupations = [o for o in self.extractor.occupations if o.skill_ids]
        progress = tqdm(range(self.num_examples), desc="Synthesizing examples", unit="ex")
        for _ in progress:
            if self.paraphraser is not None:
                progress.set_postfix(paraphrase_calls=len(self.paraphraser.cache))

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

    def _surface_form(self, skill: EscoSkill, allow_llm_paraphrase: bool = False) -> str:
        """How the skill appears in the CV: usually verbatim, sometimes an
        ESCO alt_label, and -- for matched skills only, when
        allow_llm_paraphrase is set and a paraphraser is configured --
        sometimes a genuinely differently-worded real-world equivalent.

        allow_llm_paraphrase must stay False for distractor/near-miss
        mentions: those exist specifically to teach the model to reject
        domain-plausible-but-wrong mentions, which requires them to stay
        literal/near-literal, not genuinely equivalent-but-different.

        The literal/alt_label path is gerund-converted via
        _gerundize_if_verb_led (skill.name itself, used for ground-truth
        matching, is untouched -- only what actually gets rendered into the
        CV changes). The paraphraser output already comes back as a
        noun/gerund phrase per its own prompt, so it's returned as-is.
        """
        if allow_llm_paraphrase and self.paraphraser and random.random() < synthesis.llm_paraphrase_prob:
            phrase = self.paraphraser.get(skill)
            if phrase:
                return phrase
        literal = (
            random.choice(skill.alt_labels)
            if skill.alt_labels and random.random() < synthesis.alt_label_prob
            else skill.name
        )
        return _gerundize_if_verb_led(literal)

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

        mentions = [self._surface_form(s, allow_llm_paraphrase=True) for s in matched_skills]
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
    from dotenv import load_dotenv

    load_dotenv()  # picks up OPENAI_API_KEY the same way evaluate_gemini.py/export.py load their keys

    extractor = EscoExtractor(
        skills_path=paths.raw_esco_skills,
        occupations_path=paths.raw_esco_occupations,
    )
    extractor.load()

    paraphraser = None
    if os.environ.get("OPENAI_API_KEY"):
        paraphraser = SkillParaphraser(Path(paths.paraphrase_cache))
    else:
        print(
            "⚠️  OPENAI_API_KEY not set: skipping LLM-paraphrase surface forms. "
            "Matched-skill CV evidence will be literal/ESCO-alt-label only, which "
            "reproduces the train/inference mismatch documented for cv-guestimator "
            "(rigid literal matching) -- set OPENAI_API_KEY to fix that for this run."
        )

    synthesizer = DatasetSynthesizer(extractor=extractor, num_examples=3000, paraphraser=paraphraser)
    try:
        synthesizer.build_dataset()
    finally:
        # Save whatever got cached even if the run is interrupted partway
        # through -- these are real, rate-limited API calls, not free to redo.
        if paraphraser is not None:
            paraphraser.save()

    counts: Dict[str, int] = {}
    with open(paths.processed_dataset, "w", encoding="utf-8") as f:
        for ex in synthesizer.examples:
            counts[ex.task] = counts.get(ex.task, 0) + 1
            f.write(ex.to_jsonl() + "\n")

    print(f"Wrote {len(synthesizer.examples)} examples to {paths.processed_dataset}")
    for task, count in sorted(counts.items()):
        print(f"  {task}: {count}")
    if paraphraser is not None:
        generated = sum(1 for v in paraphraser.cache.values() if v)
        print(f"  paraphrase cache: {generated}/{len(paraphraser.cache)} skills got an LLM-generated surface form")
