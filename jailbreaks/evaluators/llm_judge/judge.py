import os
import json
import re
import logging
from typing import Tuple, Any, Dict, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from google import genai
from groq import Groq

from .templates import ETHICAL_HARMFULNESS_TEMPLATE

from jailbreaks.llm import LLM

logger = logging.getLogger("LLM Judge")

# All numeric metrics that the judge is expected to emit.
# Extend this list whenever you introduce a new criterion.
METRICS = ["refusal", "harmful", "ethical", "depth"]

def parse_vote_response(text: str) -> dict:        
    """
    Parse the LLM's JSON chunk and return **all** keys it supplies,
    converting every key (and any string value) to lowercase.  
    We do *not* force the presence of any particular field – that job
    is now handled by METRICS / _empty_metrics().
    """
    match = (
        re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
        or re.search(r"({.*?})", text, re.DOTALL)
        or re.search(r"({[^}]*)", text, re.DOTALL)
    )

    if not match:
        logger.warning("Could not extract JSON from response")
        return {}

    json_string = match.group(1).strip()
    if match.re.pattern.endswith("({[^}]*)"):
        json_string += "}"  # close incomplete brace

    try:
        raw_dict = json.loads(json_string)
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse error: {exc}")
        return {}

    # ── post-processing: lowercase keys and any string values ──────────────────
    processed: dict[str, Any] = {}
    for k, v in raw_dict.items():
        new_key = str(k).lower()

        if isinstance(v, str):
            processed[new_key] = v.lower()
        elif isinstance(v, list):
            processed[new_key] = [
                elem.lower() if isinstance(elem, str) else elem for elem in v
            ]
        else:
            processed[new_key] = v

    return processed

@dataclass
class LLMJudgeVote:
    scores: Dict[str, Optional[int]] = field(default_factory=dict)

    prompt: str = ""
    response: str = ""
    llm_response: list[str] = field(default_factory=list)
    
    def metadata(self) -> dict:
        record = {
            "prompt":        self.prompt,
            "response":      self.response,
            "llm_response":  self.llm_response,
        }
        return record

class BaseLLMJudge(ABC, BaseModel):
    model: Any = Field(description="The LLM model to use for judging")
    client: Any = Field(description="Client for the LLM model")
    
    @abstractmethod
    def _score_response_fn(self, query:str, response:str) -> str:
        # processes a single prompt, getting a single response from LLM
        pass
    
    @abstractmethod
    def _score_response_fn_batch(self, prompts:list[str]) -> str:
        # processes a batch of prompts, getting a batch of responses from LLM
        pass
    
    def _prepare_prompt(self, query:str, response:str) -> str:
        return ETHICAL_HARMFULNESS_TEMPLATE.format(
            query=query, 
            response=response, 
        )

    # ---------- NEW: generic concurrent batching helper ------------------
    #
    # Sub-classes can still override this if they have a *truly* async
    # SDK, but for normal blocking SDKs this "thread-pool fan-out" gives
    # us non-blocking behaviour from the caller's point of view.
    #
    def _score_response_fn_batch(self, prompts: list[str]) -> list[str]:
        """
        Default implementation – execute `_score_response_fn` for each
        prompt in a thread-pool so the HTTP requests run in parallel.
        """
        if not prompts:
            return []

        max_workers = min(len(prompts), 16) # groq has 1000 requests per minute
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self._score_response_fn, prompts))

    def vote_batch(self, queries: list[str], responses: list[str], batch_size: int = 32) -> list[LLMJudgeVote]:
        all_votes: list[LLMJudgeVote] = []

        for i in range(0, len(queries), batch_size):
            batch_queries    = queries[i:i + batch_size]
            batch_responses  = responses[i:i + batch_size]
            formatted_prompt = [
                self._prepare_prompt(q, r) for q, r in zip(batch_queries, batch_responses)
            ]

            llm_texts  = self._score_response_fn_batch(formatted_prompt)

            vote_dicts: list[dict]      = []
            attempts:   list[list[str]] = []

            for idx, first_txt in enumerate(llm_texts):
                MAX_RESCORING_ATTEMPTS = 3
                run_texts = [first_txt]

                parsed_response = parse_vote_response(first_txt)
                if not parsed_response:
                    for _ in range(MAX_RESCORING_ATTEMPTS):
                        logger.info("Empty response detected, rescoring...")
                        prompt        = self._prepare_prompt(batch_queries[idx], batch_responses[idx])
                        rescored_txt  = self._score_response_fn(prompt)
                        run_texts.append(rescored_txt)

                        parsed_response = parse_vote_response(rescored_txt)
                        if parsed_response:
                            break

                vote_dicts.append(parsed_response or {})
                attempts.append(run_texts)

            for j, vd in enumerate(vote_dicts):
                scores = _empty_metrics()
                scores.update({k: vd.get(k) for k in METRICS if k in vd})

                vote = LLMJudgeVote(
                    scores=scores,
                    prompt=batch_queries[j],
                    response=batch_responses[j],
                    llm_response=attempts[j],
                )
                all_votes.append(vote)

        return all_votes
        
    
    # def score(self, query, response, n_votes:int=1) -> Tuple[dict, dict, list]:
    #     votes = {}
    #     key_counts = {}

    #     raw_responses = []
    #     s, raw_responses = self.vote(query, response, n_votes)
    #     for i in range(n_votes):
    #         if s[i] is not None:
    #             for key in ['refusal', 'attempt', 'useful']:
    #                 votes[key] = votes.get(key, 0) + s[i].get(key, 0)
    #                 key_counts[key] = key_counts.get(key, 0) + 1
            
    #     majority_vote = {k: v > key_counts.get(k, 0)/2 for k, v in votes.items()}
        
    #     return majority_vote, votes, key_counts, raw_responses
    
    # def vote(self, query, response, n_votes:int=1):
    #     try:
    #         formatted_prompt = self._prepare_prompt(query, response)
    #         prompts = [formatted_prompt] * n_votes
    #         text_responses = self._score_response_fn_batch(prompts)

    #         if not text_responses:
    #             logger.warning(f"No response text received from the model for query: {query[:50]}...")
    #             return None
            
    #         vote_data = [parse_vote_response(text) for text in text_responses]
            
    #         return vote_data, text_responses

    #     except Exception as e:
    #         logger.warning(f"An unexpected error occurred for query '{query[:50]}...': {e}")
    #         return None 

class GoogleLLMJudge(BaseLLMJudge):
    def __init__(self, model:str="gemini-2.0-flash-lite"):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY is not set"
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        super().__init__(model=model, client=client)
        
    def _score_response_fn(self, prompt:str) -> str:
        return self.client.models.generate_content(
            model=self.model,
            contents=prompt
        ).text.strip()
    

class LocalLLMJudge(BaseLLMJudge):
    def __init__(self, model:str):
        client = LLM(model_path=model)
        super().__init__(model=model, client=client)
    
    def _score_response_fn(self, prompt:str) -> str:
        return self._score_response_fn_batch([prompt])[0]

    def _score_response_fn_batch(self, prompts:list[str]) -> str:
        return self.client.generate_batch(prompts=prompts, max_new_tokens=200)

    class Config:
        arbitrary_types_allowed = True

class GroqLLMJudge(BaseLLMJudge):
    def __init__(self, model:str="llama-3.1-8b-instant"):
        assert os.environ.get("GROQ_API_KEY") is not None, "GROQ_API_KEY is not set"
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        super().__init__(model=model, client=client)
 
    def _score_response_fn(self, prompt:str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content

def _empty_metrics() -> dict[str, None]:
    return {m: None for m in METRICS}

