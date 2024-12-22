def evaluate_retrieval(queries, retrieved_docs, ground_truth_docs):
    if len(queries) != len(retrieved_docs) or len(queries) != len(ground_truth_docs):
        return {"error": "Mismatched lengths of queries, retrieved docs, and ground truth docs"}

    precision_scores = []
    recall_scores = []

    for retrieved, ground_truth in zip(retrieved_docs, ground_truth_docs):
        retrieved_set = set(retrieved)
        ground_truth_set = set(ground_truth)

        # True positives
        tp = len(retrieved_set & ground_truth_set)

        # Precision and recall calculations
        precision = tp / len(retrieved_set) if retrieved_set else 0
        recall = tp / len(ground_truth_set) if ground_truth_set else 0

        precision_scores.append(precision)
        recall_scores.append(recall)

    return {
        "average_precision": sum(precision_scores) / len(precision_scores),
        "average_recall": sum(recall_scores) / len(recall_scores)
    }
