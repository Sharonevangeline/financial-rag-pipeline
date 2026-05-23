import logging

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(query: str, vector_store, k: int = 5) -> list:
    """
    Retrieve the k most semantically similar chunks for a given query.
    Returns list of (document, score) tuples.
    """
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    logger.info("Retrieved %d chunks for query: %s", len(results), query[:50])
    return results


def format_context(results: list) -> tuple[str, list]:
    """
    Format retrieved chunks into a single context string for the LLM.
    Also returns source references for display.
    """
    context_parts = []
    sources = []

    for i, (doc, score) in enumerate(results):
        company   = doc.metadata.get("company", "Unknown")
        page      = doc.metadata.get("page", "?")
        source    = f"{company} — Page {page + 1}"

        context_parts.append(
            f"[Source {i + 1}: {source}]\n{doc.page_content}"
        )
        sources.append({
            "index": i + 1,
            "company": company,
            "page": page + 1,
            "score": round(score, 3),
            "preview": doc.page_content[:150] + "...",
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources