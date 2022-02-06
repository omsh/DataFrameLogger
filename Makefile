install:
	pip install --upgrade pip && pip install -e .[dev]

test:
	python -m pytest test_*.py --doctest-modules --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html

format:
	black *.py

lint:
	pylint --disable=R,C logger.py

all: install format test
