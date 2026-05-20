# Industrial RAG Assistant

산업 문서(PDF)를 업로드하면 자연어 질문으로 검색·답변을 받을 수 있는 RAG 기반 AI 서비스입니다.

## Tech Stack

- **Backend**: FastAPI
- **RAG**: LangChain + FAISS
- **LLM**: OpenAI GPT-4o-mini
- **Embedding**: text-embedding-3-small
- **Container**: Docker

## 실행 방법

### 1. 환경 설정

```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. PDF 문서 추가

`data/sample_docs/` 폴더에 PDF 파일을 넣습니다.

### 4. 서버 실행

```bash
uvicorn app.main:app --reload
```

### 5. 문서 임베딩 (최초 1회)

```bash
curl -X POST http://localhost:8000/ingest
```

### 6. 질문

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "모터 전력 소비 패턴은 어떻게 되나요?", "top_k": 3}'
```

## Docker 실행

```bash
docker build -t industrial-rag-assistant .
docker run -p 8000:8000 --env-file .env industrial-rag-assistant
```

## API 문서

서버 실행 후 `http://localhost:8000/docs` 에서 Swagger UI로 확인 가능합니다.

## Architecture

```
PDF 문서 → 텍스트 추출(pypdf) → 청크 분할 → 임베딩 → FAISS 저장
질문 → 유사 청크 검색 → LLM 프롬프트 구성 → 답변 생성
```
