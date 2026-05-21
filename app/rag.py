import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from app.config import settings


_SYSTEM_PROMPT = (
    "You are an expert industrial document analyst. "
    "The context below is extracted from industrial PDF documents (likely in English). "
    "Answer the user's question in Korean using information from the context. "
    "Extract and synthesize relevant technical information even if the question and context languages differ. "
    "Only if the context is completely unrelated to the question, say: "
    "\"해당 문서에서 관련 내용을 찾을 수 없습니다.\"\n"
    "Be concise and technical."
)


def _get_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return FAISS.load_local(
        settings.vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def query(question: str, top_k: int = 5) -> dict:
    start = time.time()
    vectorstore = _get_vectorstore()

    results = vectorstore.similarity_search_with_score(question, k=top_k)
    elapsed = int((time.time() - start) * 1000)

    if not results:
        return {
            "answer": "해당 문서에서 관련 내용을 찾을 수 없습니다.",
            "sources": [],
            "latency_ms": elapsed,
            "retrieved_chunks": 0,
        }

    context_parts = []
    sources = []
    seen: set = set()
    for doc, score in results:
        context_parts.append(doc.page_content)
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": src.split("/")[-1],
                "page": (page + 1) if page is not None else None,
                "score": round(float(score), 4),
            })

    context = "\n\n---\n\n".join(context_parts)

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ])

    return {
        "answer": response.content,
        "sources": sources,
        "latency_ms": int((time.time() - start) * 1000),
        "retrieved_chunks": len(results),
    }
