.PHONY: install run test lint docker-build docker-run clean

install:
	pip install -r requirements.txt
	pip install pytest flake8

run:
	streamlit run app.py

test:
	pytest tests/

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

docker-build:
	docker build -t antientropy .

docker-run:
	docker run -p 8501:8501 antientropy

docker-compose-up:
	docker-compose up

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
