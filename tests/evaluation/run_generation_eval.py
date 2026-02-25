"""
Generation Evaluation Runner.

Runs all queries through the full RAG pipeline (retrieval + generation),
then scores with RAGAS metrics: Faithfulness, Answer Relevancy,
Context Precision, Context Recall, Answer Correctness.

Run full evaluation:
    python -m tests.evaluation.run_generation_eval

Print cached results (no API calls):
    python -m tests.evaluation.run_generation_eval --cached
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv
load_dotenv()

from evaluation.generation_eval import create_evaluator, GenerationEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def run_evaluation():
    """Run the full generation evaluation and print results."""
    evaluator = await create_evaluator()
    await evaluator.evaluate()


def print_cached():
    """Print the most recent saved results without re-running."""
    data = GenerationEvaluator.load_latest_results()
    if data is None:
        print("No cached results found. Run the full evaluation first.")
        sys.exit(1)
    GenerationEvaluator.print_saved_results(data)


if __name__ == "__main__":
    if "--cached" in sys.argv:
        print_cached()
    else:
        asyncio.run(run_evaluation())
