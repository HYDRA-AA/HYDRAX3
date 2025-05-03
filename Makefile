.PHONY: help clean test lint format doctest docs install dev-install

help:
	@echo "Available commands:"
	@echo "  clean        - Remove build artifacts and cache files"
	@echo "  test         - Run the test suite"
	@echo "  lint         - Check code style with flake8"
	@echo "  format       - Format code with black and isort"
	@echo "  doctest      - Run doctests"
	@echo "  docs         - Generate documentation"
	@echo "  install      - Install the package"
	@echo "  dev-install  - Install the package in development mode"
	@echo "  vectors      - Generate test vectors"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name ".pytest_cache" -delete
	find . -name "*.egg" -delete
	rm -rf docs/_build/

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ tests/ --exclude=__pycache__,*.egg-info

format:
	black src/ tests/ tools/
	isort src/ tests/ tools/

doctest:
	python -m pytest --doctest-modules src/

docs:
	cd docs && make html

install:
	pip install .

dev-install:
	pip install -e ".[dev]"

vectors:
	python tools/generate_test_vectors.py
