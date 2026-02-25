"""
Retrieval Evaluation Runner.

Run:
    python -m tests.evaluation.run_retrieval_eval
"""

import asyncio
import logging

from dotenv import load_dotenv
load_dotenv()

from evaluation.retrieval_eval import create_evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def run_evaluation():
    """Run the full retrieval evaluation and print results."""
    evaluator = await create_evaluator()
    result = await evaluator.evaluate(top_k=10)
    print(result.summary())


if __name__ == "__main__":
    asyncio.run(run_evaluation())
