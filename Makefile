DOCKER_COMPOSE = docker compose
MODEL_CACHE_DIR = ${HOME}/.cache/huggingface
COMPOSE_FILE = docker-compose.yml


.PHONY: run stop clean-cache logs rebuild

run:
	$(DOCKER_COMPOSE) up -d

stop:
	$(DOCKER_COMPOSE) down

clean-cache:
	rm -rf $(MODEL_CACHE_DIR)/*

logs:
	$(DOCKER_COMPOSE) logs -f

rebuild:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build --no-cache
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down
	$(DOCKER_COMPOSE)  -f $(COMPOSE_FILE) up -d