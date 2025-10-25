# Static-Embedding-Japanese-FastAPI

OpenAI compatible embedding API that uses [hotchpotch/static-embedding-japanese](https://huggingface.co/hotchpotch/static-embedding-japanese) for embeddings

## Install

There are 2 options to install: Docker or local Python install.

### Install (Docker)

Run the API locally using Docker:

```bash
docker run -p 8080:8080 -d ghcr.io/kun432/static-embedding-japanese-fastapi:v0.0.2
```

Also, you can set the `DIM` environment variable to change the embedding dimension.

```bash
docker run -e DIM=128 -p 8080:8080 -d ghcr.io/kun432/static-embedding-japanese-fastapi:v0.0.2
```

### Install (Local python)

Install and run the API server locally using Python. uv is required.

Clone the repo:

```bash
git clone https://github.com/kun432/static-embedding-japanese-fastapi
cd static-embedding-japanese-fastapi
```

Install dependencies:

```bash
uv sync
```

Run the webserver:

```bash
uv run -- uvicorn main:app --port 8080 --reload
```

Also, you can set the `DIM` environment variable to change the embedding dimension.

```bash
DIM=128 uv run -- uvicorn main:app --port 8080 --reload
```

## Usage

After you've installed, you can visit the API docs on [http://localhost:8080/docs](http://localhost:8080/docs)

You can also use CURL to get embeddings:

For a single string,

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your text string goes here",
    "model": "hotchpotch/static-embedding-japanese"
  }'
```

For a list of strings,

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Your text string goes here",
      "あなたのテキスト文字列がここに入ります。"
    ],
    "model": "hotchpotch/static-embedding-japanese"
  }'
```

Even the OpenAI Python client can be used to get embeddings:

```python
from openai import OpenAI

client = OpenAI(
    base_url = "http://localhost:8080/v1"
    api_key = "this isn't used but openai client requires it"
)

response = client.embeddings.create(
  input="Some text",
  model="hotchpotch/static-embedding-japanese"
)
print(response.data[0].embedding)
```

## Supported Models

`hotchpotch/static-embedding-japanese` is the only model supported. ()

If you want to use different models, you should check:

- [substratusai/stapi](https://github.com/substratusai/stapi)
- [michaelfeil/infinity](https://github.com/michaelfeil/infinity)

## Develop

Clone the repo:

```bash
git clone https://github.com/kun432/static-embedding-japanese-fastapi
cd static-embedding-japanese-fastapi
```

Install dependencies:

```bash
uv sync
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## TODO

- [ ] remove requirement.txt so that dependencies are managed by uv only

## Acknowledgments

This project is based on and extends ["STAPI: Sentence Transformers API"](https://github.com/substratusai/stapi) by Sam Stoelinga aka Samos123 & Nick Stogner released under the Apache License 2.0.
I would like to thank Sam & Nick for creating and maintaining the original codebase, and for granting permission to use, modify, and redistribute it under the terms of the Apache License 2.0.
All significant changes and additions in this repository were implemented by kun432, and are likewise released under the Apache License 2.0.

In addition to the codebase, Docker image bundles Sentence Transformers model ["hotchpotch/static-embedding-japanese"](https://huggingface.co/hotchpotch/static-embedding-japanese) by Yuichi Tateno aka hotchpotch released under the MIT License.
I would like to thank hotchpotch for creating and maintaining the original model, and for granting permission to use, modify, and redistribute it under the terms of the MIT License.
