import os
import json
import re
import logging
from typing import Tuple
from pydantic import BaseModel

from google import genai

from .samples import examples
from .templates import REFUSAL_QUALITY_TEMPLATE

logger = logging.getLogger("LLM Judge")

def parse_score_response(text:str) -> dict|None:        
    match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"({.*?})", text, re.DOTALL)

    if not match:
        logger.warning(f"Warning: Could not extract JSON from response for query: {text[:50]}...")
        return None

    json_string = match.group(1).strip()
    try:
        score_data = json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"Warning: Could not parse JSON from response for query: {text[:50]}...")
        logger.warning(f"Error: {e}")
        return None

    if 'attempt' not in score_data or 'useful' not in score_data or 'refusal' not in score_data:
        logger.warning(f"Warning: Missing keys in JSON response for query: {text[:50]}...")
        logger.warning(f"Parsed JSON: {score_data}")
        return None

    return score_data

class LLMJudge(BaseModel):
    def __init__(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY is not set"
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        model = "gemini-2.0-flash-lite"
        
        self._score_response_fn = lambda query, response: client.models.generate_content(
            model=model,
            contents=REFUSAL_QUALITY_TEMPLATE.format(
                query=query, 
                response=response, 
                examples=json.dumps(examples, indent=4)
            )
        ).text.strip()
    
    def score(self, query, response, n_votes:int=1) -> Tuple[dict, dict, list]:
        votes = {}
        raw_responses = []
        for _ in range(n_votes):
            s, text = self.vote(query, response)
            if s is not None:
                for k, v in s.items():
                    votes[k] = votes.get(k, 0) + v
                raw_responses.append(text)
        majority_vote = {k: v > n_votes/2 for k, v in votes.items()}
        
        return majority_vote, votes, raw_responses
    
    def vote(self, query, response):
        try:
            text = self._score_response_fn(query, response)

            if not text:
                logger.warning(f"No response text received from the model for query: {query[:50]}...")
                return None
            
            score_data = parse_score_response(text)

            return score_data, text

        except Exception as e:
            logger.warning(f"An unexpected error occurred for query '{query[:50]}...': {e}")
            return None 
