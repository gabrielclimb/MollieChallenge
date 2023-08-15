FROM python:3.11 as base

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./src /app/src

FROM base as api
CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base as train
CMD ["python", "-m", "src.train.model.train"]
