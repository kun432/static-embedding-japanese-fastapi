import importlib
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# テスト対象モジュールを後で動的に import するために関数化する
def get_app_with_dim(dim: str | None = None) -> FastAPI:
    if dim is not None:
        os.environ["DIM"] = dim
    else:
        os.environ.pop("DIM", None)
    # リロードして環境変数を反映
    module = importlib.import_module("main")
    importlib.reload(module)
    return module.app


def test_read_healthz() -> None:
    """/healthz エンドポイントがステータス 200 を返すことを検証"""
    app = get_app_with_dim()
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200


def test_embedding_str() -> None:
    """文字列入力 1 件で埋め込みが生成されることを検証"""
    app = get_app_with_dim()
    with TestClient(app) as client:
        embedding_request = {
            "input": "substratus.ai has some great LLM OSS projects for K8s",
        }
        response = client.post("/v1/embeddings", json=embedding_request)
        assert response.status_code == 200
        embedding_response = response.json()
        assert isinstance(embedding_response["data"], list)
        assert isinstance(embedding_response["data"][0]["embedding"], list)
        assert isinstance(embedding_response["data"][0]["embedding"][0], float)


def test_embedding_list_str() -> None:
    """文字列リスト入力で複数埋め込みが生成され、かつ内容が異なることを検証"""
    app = get_app_with_dim()
    with TestClient(app) as client:
        embedding_request = {
            "input": [
                "substratus.ai has some great LLM OSS projects for K8s",
                "2nd string",
            ],
        }
        response = client.post("/v1/embeddings", json=embedding_request)
        assert response.status_code == 200
        embedding_response = response.json()
        assert isinstance(embedding_response["data"], list)
        assert isinstance(embedding_response["data"][0]["embedding"], list)
        assert isinstance(embedding_response["data"][0]["embedding"][0], float)
        assert isinstance(embedding_response["data"][1]["embedding"], list)
        assert isinstance(embedding_response["data"][1]["embedding"][0], float)
        assert embedding_response["data"][0]["embedding"] != embedding_response["data"][1]["embedding"]


def test_invalid_dim_env() -> None:
    """DIM環境変数に不正値を指定した場合は RuntimeError が発生することを検証"""
    with pytest.raises(RuntimeError):
        get_app_with_dim("invalid_dim")
