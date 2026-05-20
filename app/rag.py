from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.config import settings


PROMPT_TEMPLATE = """당신은 산업 문서를 분석하는 AI 어시스턴트입니다.
아래 컨텍스트를 바탕으로 질문에 정확하게 답하세요.
컨텍스트에 없는 내용은 '해당 문서에서 찾을 수 없습니다'라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""


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


def query(question: str, top_k: int = 3) -> dict:
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    result = chain.invoke({"query": question})
    sources = list({doc.metadata.get("source", "unknown") for doc in result["source_documents"]})

    return {"answer": result["result"], "sources": sources}
