.PHONY: setup install test train api clean lint format

# Setup project
setup:
	python scripts/setup_environment.py

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v

# Train model
train:
	python src/main.py --mode train

# Start API
api:
	python src/main.py --mode api

# Start real-time processing
realtime:
	python src/main.py --mode realtime

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/

# Lint code
lint:
	flake8 src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Download datasets
data:
	python scripts/download_datasets.py

# Run all checks
check: lint test

# Docker build
docker-build:
	docker-compose build

# Docker run
docker-run:
	docker-compose up
