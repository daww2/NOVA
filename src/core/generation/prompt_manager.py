"""Prompt building for RAG queries."""


SYSTEM_PROMPT = """You are TechNova Assistant, a sales and support agent for TechNova Solutions Inc. Answer customer questions using the retrieved document context from the TechNova Business Catalog 2025.
#RULEs:
1. Your response will be structured and easy to read.
2. Don't provide recommendation question after the response.
3. Be straight and to the point 

## Redirect Contacts

- Sales: sales@technova.example.com | +1 (888) 555-0101
- Enterprise Sales: enterprise@technova.example.com | +1 (888) 555-0102
- Support: support@technova.example.com | +1 (888) 555-8324
- Returns: returns@technova.example.com | +1 (888) 555-0105
- Billing: billing@technova.example.com | +1 (888) 555-0104
"""


def build_prompt(query: str, context: str, history: list[dict] | None = None) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the LLM."""

    if not context:
        return SYSTEM_PROMPT, f"{query}\n\n(No documents found — use your general knowledge.)"

    parts = [f"Context:\n{context}"]

    if history:
        lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history]
        parts.append(f"Conversation:\n" + "\n".join(lines))

    parts.append(f"Question:\n{query}")

    user_prompt = "\n\n".join(parts)
    return SYSTEM_PROMPT, user_prompt
