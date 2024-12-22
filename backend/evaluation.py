from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def evaluate_llm_output(responses, ground_truths):
    if len(responses) != len(ground_truths):
        return {"error": "Responses and ground truths must have the same length"}

    rouge = Rouge()
    bleu_scores = []
    rouge_scores = []

    for response, ground_truth in zip(responses, ground_truths):
        bleu_scores.append(sentence_bleu([ground_truth.split()], response.split()))
        rouge_result = rouge.get_scores(response, ground_truth)
        rouge_scores.append(rouge_result[0]['rouge-l']['f'])

    return {
        "average_bleu": sum(bleu_scores) / len(bleu_scores),
        "average_rouge": sum(rouge_scores) / len(rouge_scores)
    }
