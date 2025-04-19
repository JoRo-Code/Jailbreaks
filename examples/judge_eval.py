import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from jailbreaks.evaluators.llm_judge.judge import (
    GroqLLMJudge,
    GoogleLLMJudge,
    LocalLLMJudge
)
from jailbreaks.evaluators.llm_judge.samples import test_samples

logger = logging.getLogger("LLM Judge")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # judge = LocalLLMJudge(
    #     #model="Qwen/Qwen2-0.5B-Instruct"
    #     model="Qwen/Qwen2.5-7B-Instruct"
    # )
    
    judge = GroqLLMJudge(model="llama-3.1-8b-instant")
    # judge = GoogleLLMJudge(model="gemini-2.0-flash-lite")
    #judge = GoogleLLMJudge(model="gemini-1.5-flash")
        
    results = []
    total_evaluated = 0
    
    refusal_correct = 0
    attempt_correct = 0
    useful_correct = 0
    
    logger.info("Starting evaluation...")
    
    n_votes_per_sample = 3
    
    # Collect multiple votes per sample
    all_votes = []
    for i in range(n_votes_per_sample):
        logger.info(f"Collecting vote set {i+1}/{n_votes_per_sample}...")
        votes = judge.vote_batch(
            [gen_response["prompt"] for gen_response in test_samples],
            [gen_response["response"] for gen_response in test_samples]
        )
        all_votes.append(votes)
    
    # Organize votes by sample
    sample_votes = [[] for _ in range(len(test_samples))]
    for vote_set in all_votes:
        for i, vote in enumerate(vote_set):
            sample_votes[i].append(vote)
    
    # Process each sample with its multiple votes
    for i, votes_for_sample in enumerate(sample_votes):
        assert all(vote.prompt == test_samples[i]["prompt"] for vote in votes_for_sample), f"Prompt mismatch for sample {i+1}"
        logger.info(f"\n--- Evaluating Sample {i+1} ---")
        logger.info(f"Prompt: {votes_for_sample[0].prompt[:80]}...")
        
        # Determine final classification by majority voting
        attempts = [vote.attempt for vote in votes_for_sample]
        usefuls = [vote.useful for vote in votes_for_sample]
        refusals = [vote.refusal for vote in votes_for_sample]
        
        # Filter out None values when finding most common response
        def most_common_non_none(values):
            filtered = [v for v in values if v is not None]
            if not filtered:
                return None
            return max(set(filtered), key=filtered.count)
            
        final_attempt = most_common_non_none(attempts)
        final_useful = most_common_non_none(usefuls)
        final_refusal = most_common_non_none(refusals)
                    
        total_evaluated += 1
        is_attempt_correct = (test_samples[i]["attempt"] == final_attempt)
        is_useful_correct = (test_samples[i]["useful"] == final_useful)
        is_refusal_correct = (test_samples[i]["refusal"] == final_refusal)
        if is_attempt_correct:
            attempt_correct += 1
        if is_useful_correct:
            useful_correct += 1
        if is_refusal_correct:
            refusal_correct += 1
            
        results.append({
            "prompt": votes_for_sample[0].prompt,
            "response": votes_for_sample[0].response,
            "LLM_responses": [vote.llm_response for vote in votes_for_sample],
            "votes": [{
                "attempt": vote.attempt,
                "useful": vote.useful,
                "refusal": vote.refusal
            } for vote in votes_for_sample],
            "truth_attempt": test_samples[i]["attempt"],
            "truth_useful": test_samples[i]["useful"],
            "truth_refusal": test_samples[i]["refusal"],
            "pred_attempt": final_attempt,
            "pred_useful": final_useful,
            "pred_refusal": final_refusal,
            "is_attempt_correct": is_attempt_correct,
            "is_useful_correct": is_useful_correct,
            "is_refusal_correct": is_refusal_correct,
        })
        
    # Calculate summary metrics before writing to file
    if total_evaluated > 0:
        attempt_accuracy = (attempt_correct / total_evaluated) * 100
        useful_accuracy = (useful_correct / total_evaluated) * 100
        refusal_accuracy = (refusal_correct / total_evaluated) * 100
        
        # Add overview section to the results
        overview = {
            "total_samples": total_evaluated,
            "refusal_accuracy": round(refusal_accuracy, 2),
            "refusal_correct": refusal_correct,
            "attempt_accuracy": round(attempt_accuracy, 2),
            "attempt_correct": attempt_correct,
            "useful_accuracy": round(useful_accuracy, 2),
            "useful_correct": useful_correct,
            "model": judge.model,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create final output with overview and detailed results
        output = {
            "overview": overview,
            "detailed_results": results
        }
    else:
        output = {
            "overview": {"error": "No samples were successfully evaluated."},
            "detailed_results": []
        }
    
    output_dir = "llm_judge_results"
    os.makedirs(output_dir, exist_ok=True)
    results_filename = f"{output_dir}/{judge.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_filename, 'w') as f:
            json.dump(output, f, indent=4)
        logger.info(f"\nEvaluation results saved to {results_filename}")
    except Exception as e:
        logger.error(f"\nError saving results to {results_filename}: {e}")
        
    

    logger.info("\n--- Evaluation Summary ---")
    if total_evaluated > 0:
        attempt_accuracy = (attempt_correct / total_evaluated) * 100
        useful_accuracy = (useful_correct / total_evaluated) * 100
        refusal_accuracy = (refusal_correct / total_evaluated) * 100
        logger.info(f"Total samples evaluated: {total_evaluated}")
        logger.info(f"Refusal Classification Accuracy: {refusal_accuracy:.2f}% ({refusal_correct}/{total_evaluated})")
        logger.info(f"Attempt Classification Accuracy: {attempt_accuracy:.2f}% ({attempt_correct}/{total_evaluated})")
        logger.info(f"Useful Classification Accuracy: {useful_accuracy:.2f}% ({useful_correct}/{total_evaluated})")
    else:
        logger.info("No samples were successfully evaluated.")
    