import logging
import os
import platform
from contextlib import asynccontextmanager

# Apple Silicon (MPS) で未実装演算が発生した際に CPU にフォールバック
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# モデルは固定
MODEL_NAME = "hotchpotch/static-embedding-japanese"
# 埋め込み次元数 (1024 = デフォルト)
SUPPORTED_DIMS = {32, 64, 128, 256, 512, 1024}
raw_dim = os.getenv("DIM", "1024")
try:
    DIM = int(raw_dim)
except ValueError as e:
    raise RuntimeError(
        f"Invalid DIM={raw_dim!r}. Must be an integer. Supported values: {sorted(SUPPORTED_DIMS)}"
    ) from e
if DIM not in SUPPORTED_DIMS:
    raise RuntimeError(f"Invalid DIM={DIM}. Supported values: {sorted(SUPPORTED_DIMS)}")
model_name = MODEL_NAME
model: SentenceTransformer | None = None

logger.info(f"Loading model '{model_name}' with DIM={DIM}")


class EmbeddingRequest(BaseModel):
    input: str | list[str] = Field(examples=["substratus.ai provides the best LLM tools"])
    model: str = Field(
        examples=[model_name],
        default=model_name,
    )


class EmbeddingData(BaseModel):
    embedding: list[float]
    index: int
    object: str


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingData]
    model: str
    usage: Usage
    object: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:  # noqa: RUF029
    global model
    model = SentenceTransformer(model_name, truncate_dim=DIM)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/v1/embeddings")
async def embedding(item: EmbeddingRequest) -> EmbeddingResponse:
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="model not loaded")
    if isinstance(item.input, str):
        vectors = await run_in_threadpool(
            model.encode,
            item.input,
            show_progress_bar=False,
        )
        tokens = len(vectors)
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=vectors, index=0, object="embedding")],
            model=model_name,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
            object="list",
        )
    if isinstance(item.input, list):
        if not item.input:
            return EmbeddingResponse(
                data=[],
                model=model_name,
                usage=Usage(prompt_tokens=0, total_tokens=0),
                object="list",
            )

        texts: list[str] = []
        for text_input in item.input:
            if not isinstance(text_input, str):
                raise HTTPException(
                    status_code=400,
                    detail="input needs to be an array of strings or a string",
                )
            texts.append(text_input)

        raw_embeddings = await run_in_threadpool(
            model.encode,
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if getattr(raw_embeddings, "ndim", 1) == 1:
            vectors_list = [raw_embeddings.tolist()]
        else:
            vectors_list = raw_embeddings.tolist()

        embeddings = []
        tokens = 0
        for index, vectors in enumerate(vectors_list):
            tokens += len(vectors)
            embeddings.append(EmbeddingData(embedding=vectors, index=index, object="embedding"))
        return EmbeddingResponse(
            data=embeddings,
            model=model_name,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
            object="list",
        )
    raise HTTPException(status_code=400, detail="input needs to be an array of strings or a string")


@app.get("/")
@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
