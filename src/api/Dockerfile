FROM python:3.9

WORKDIR /code

COPY . /code/app

RUN pip install "fastapi[standard]" pandas mlflow

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
