import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime

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
    
    N_VOTES = 3
    
    results = []
    total_evaluated = 0
    
    refusal_correct = 0
    attempt_correct = 0
    useful_correct = 0
    
    logger.info("Starting evaluation...")
    for i, gen_response in enumerate(test_samples):
        logger.info(f"\n--- Evaluating Sample {i+1} ---")
        logger.info(f"Prompt: {gen_response['prompt'][:80]}...")
        
        majority_vote, votes, n_votes, raw_votes = judge.score(gen_response["prompt"], gen_response["response"], n_votes=N_VOTES)
        
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
            "votes": votes,
            "n_votes": n_votes
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
        
    results_filename = f"llm_judge_results_{judge.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    