mlflow:
	docker-compose -f infra/docker-compose.yml up --build

mlflow-down:
	docker-compose -f infra/docker-compose.yml down

api:
	docker-compose up --build api

train:
	docker-compose up --build train

down:
	docker-compose down
