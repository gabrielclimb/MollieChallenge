detached ?= false

mlflow:
ifeq ($(detached), true)
	docker-compose -f infra/docker-compose.yml up -d --build
else
	docker-compose -f infra/docker-compose.yml up --build
endif

mlflow-down:
	docker-compose -f infra/docker-compose.yml down

api:
ifeq ($(detached), true)
	docker-compose up --build -d api
else
	docker-compose up --build api
endif

train:
ifeq ($(detached), true)
	docker-compose up --build -d train
else
	docker-compose up --build train
endif


down:
	docker-compose -f infra/docker-compose.yml down && docker-compose down
