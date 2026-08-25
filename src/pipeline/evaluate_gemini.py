# src/evaluate_gemini.py

import os
import time
from google.genai.errors import APIError
from google import genai
from dotenv import load_dotenv

from src.core.schema import (
    JobRequirementsOutput,
    OverallExperienceResponse,
    SkillEvaluationDecision,
)
from src.pipeline.evaluate import JsonEvaluator

load_dotenv()

class GeminiRunner:
    """Handles generating completions via the Gemini 3.1 Flash-Lite API."""

    def __init__(self):
        # Automatically picks up GEMINI_API_KEY from your environment variables
        self.client = genai.Client()
        self.model_name = "gemini-3.1-flash-lite"

    @staticmethod
    def _schema_for_prompt(prompt: str):
        """Picks the response schema for the agent task, recognised by the
        system prompt the evaluator prepends to the plain-text prompt."""
        if prompt.startswith("Extract atomic technical and operational skill_names"):
            return JobRequirementsOutput
        if prompt.startswith("Extract and classify the candidate's professional work experience"):
            return OverallExperienceResponse
        return SkillEvaluationDecision

    def generate(self, prompt: str, max_retries: int = 3) -> str:
            """Sends the prompt to Gemini, enforces the Pydantic schema, and handles rate limits."""
            for attempt in range(max_retries):
                try:
                    # 1. Throttle requests to stay under the free-tier 15 RPM limit
                    time.sleep(4)

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            # Enforce exact Pydantic schema structure
                            "response_schema": self._schema_for_prompt(prompt),
                        },
                    )
                    return response.text or "{}"

                except APIError as e:
                    # Handle temporary HTTP 429 / RESOURCE_EXHAUSTED rate limits
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = 10 * (attempt + 1)
                        print(f"\n⚠️ Rate limit hit (429). Waiting {wait_time}s before retry (Attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n❌ API Error: {e}")
                        return "{}"
                except Exception as e:
                    print(f"\n❌ Unexpected Error: {e}")
                    return "{}"

            return "{}"


class GeminiEvaluator(JsonEvaluator):
    """Extends your existing JsonEvaluator but overrides the local model."""
    
    def __init__(self):
        
        # 1. Let the parent JsonEvaluator set up paths and dataset loading
        super().__init__()
        # Completely bypass the Unsloth ModelRunner initialization
        # and inject our Gemini API runner instead
        self.runner = GeminiRunner()


def run() -> None:
    """Standalone run entry point for Gemini API evaluation."""
    
    # Ensure the API key is present
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable is missing.")
        print("Please run: $env:GEMINI_API_KEY='your_key' (Windows) before executing.")
        return

    print(f"\n🚀 Booting up Gemini 3.1 Flash-Lite Evaluation Pipeline...")
    
    evaluator = GeminiEvaluator()
    
    # Leverages your exact test.jsonl configured in paths.test_file
    # and generates the exact same matplotlib charts!
    evaluator.evaluate_dataset(num_samples=50)


if __name__ == "__main__":
    run()