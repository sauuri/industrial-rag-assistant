from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.config import settings


PROMPT_TEMPLATE = """You are an AI assistant that analyzes industrial documents.
The context below may be in English. Answer the question in Korean based on the context.
If the context does not contain relevant information, say "해당 문서에서 관련 내용을 찾을 수 없습니다."

Context:
{context}

Question: {question}

Answer in Korean:"""


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
