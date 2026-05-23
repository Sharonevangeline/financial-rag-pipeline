import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def build_prompt(question: str, context: str) -> str:
    return f"""You are a financial analyst assistant helping users understand
corporate annual reports.

Use ONLY the context below to answer the question. If the answer is not in 
the context, say "I could not find this information in the provided documents."
Always cite which company and page number your answer comes from.

Context:
{context}

Question: {question}

Answer:"""


def get_answer_ollama(question: str, context: str) -> str:
    """Generate answer using Mistral via Ollama (local)."""
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=os.getenv("LLM_MODEL", "mistral"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
        )
        prompt = build_prompt(question, context)
        return llm.invoke(prompt)
    except Exception as e:
        logger.error("Ollama error: %s", e)
        raise


def get_answer_groq(question: str, context: str) -> str:
    """Generate answer using Llama3 via Groq API (for deployment)."""
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = build_prompt(question, context)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Groq error: %s", e)
        raise


def get_answer(question: str, context: str) -> str:
    """
    Route to Groq if USE_GROQ=true, otherwise use Ollama.
    Allows same codebase for local dev and cloud deployment.
    """
    use_groq = os.getenv("USE_GROQ", "false").lower() == "true"
    if use_groq:
        return get_answer_groq(question, context)
    return get_answer_ollama(question, context)