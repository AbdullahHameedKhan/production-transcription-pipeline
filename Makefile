.PHONY: install test lint run docker clean

install:
	pip install -e ".[whisper,api,dev]"

test:
	pytest -v

lint:
	ruff check src tests
	mypy src

run:
	uvicorn transcription.api.main:app --reload

docker:
	docker build -t transcription-pipeline .
	docker run -p 8000:8000 transcription-pipeline

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +