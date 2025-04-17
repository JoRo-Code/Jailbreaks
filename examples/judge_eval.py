import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging

from jailbreaks.evaluators.llm_judge.judge import LLMJudge
from jailbreaks.evaluators.llm_judge.samples import test_samples

logger = logging.getLogger("LLM Judge")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    judge = LLMJudge()
    
    results = []
    total_evaluated = 0
    
    refusal_correct = 0
    attempt_correct = 0
    useful_correct = 0
    
    logger.info("Starting evaluation...")
    for i, gen_response in enumerate(test_samples):
        logger.info(f"\n--- Evaluating Sample {i+1} ---")
        logger.info(f"Prompt: {gen_response['prompt'][:80]}...")
        
        majority_vote, votes, raw_votes = judge.score(gen_response["prompt"], gen_response["response"], n_votes=3)
        
        s = majority_vote
        
        total_evaluated += 1
        is_attempt_correct = (gen_response["attempt"] == s.get("attempt"))
        is_useful_correct = (gen_response["useful"] == s.get("useful"))
        is_refusal_correct = (gen_response["refusal"] == s.get("refusal"))
        if is_attempt_correct:
            attempt_correct += 1
        if is_useful_correct:
            useful_correct += 1
        if is_refusal_correct:
            refusal_correct += 1
            
        results.append({
            "prompt": gen_response["prompt"],
            "response": gen_response["response"],
            "LLM_responses": raw_votes,
            "truth_attempt": gen_response["attempt"],
            "truth_useful": gen_response["useful"],
            "truth_refusal": gen_response["refusal"],
            "pred_attempt": s.get("attempt"),
            "pred_useful": s.get("useful"),
            "pred_refusal": s.get("refusal"),
            "is_attempt_correct": is_attempt_correct,
            "is_useful_correct": is_useful_correct,
            "is_refusal_correct": is_refusal_correct,
            "votes": votes
        })
        
    results_filename = "llm_judge_results.json"
    try:
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"\nEvaluation results saved to {results_filename}")
    except IOError as e:
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
    