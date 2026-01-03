import json
import time
from QA.answer_generator import answer_question
from Vector_store.retriever import Retriever
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_evaluation():
    with open("evaluation/benchmark_questions.json") as f:
        questions = json.load(f)

    retriever = Retriever(
        "data/embeddings/vector.index",
        "data/embeddings/metadata.pkl",
        top_k=5
    )

    results = []

    for q in questions:
        start = time.time()

        retrieved = retriever.retrieve(q["question"])
        answer, citations = answer_question(q["question"])

        latency = round(time.time() - start, 2)

        modalities = list(set([r["modality"] for r in retrieved]))

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "expected_modality": q["expected_modality"],
            "retrieved_modalities": modalities,
            "latency_sec": latency,
            "has_citation": len(citations) > 0
        })

    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Evaluation completed")
    for r in results:
        print(
            f"Q{r['question_id']} | "
            f"Modalities: {r['retrieved_modalities']} | "
            f"Latency: {r['latency_sec']}s"
        )

if __name__ == "__main__":
    run_evaluation()
