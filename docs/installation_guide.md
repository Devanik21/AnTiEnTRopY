# Installation Guide

## Prerequisites
- Python 3.9+
- pip (Python package manager)

## Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Devanik21/AnTiEnTRopY.git
   cd AnTiEnTRopY
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Using Docker
A `docker-compose.yml` is provided for containerized deployment:
```bash
docker-compose up --build
```
