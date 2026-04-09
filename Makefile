DOCKER_COMPOSE = docker compose
MODEL_CACHE_DIR = ${HOME}/.cache/huggingface
COMPOSE_FILE = docker-compose.yml


.PHONY: run stop clean-cache logs rebuild wipe

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

wipe:
	$(DOCKER_COMPOSE) down -v

# Makefile кроссплатформенный (Linux + Windows PowerShell)
# Используется FORCE=yes для перезаписи существующих файлов

# Кроссплатформенный Makefile для инициализации файлов из шаблонов
# Используется FORCE=yes для перезаписи существующих файлов

init-env:
init-env:
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -ExecutionPolicy Bypass -File init-env.ps1 $(if $(FORCE),-Force)
else
	@find . -type f -name "*.example*" | while read file; do \
		dir=$$(dirname "$$file"); \
		base=$$(basename "$$file"); \
		if echo "$$base" | grep -q "\.example\.env$$"; then \
			target=$$(echo "$$base" | sed 's/\.example\.env$$/.env/'); \
		elif echo "$$base" | grep -q "\.example\.json$$"; then \
			target=$$(echo "$$base" | sed 's/\.example\.json$$/.json/'); \
		elif echo "$$base" | grep -q "\.example\..+"; then \
			target=$$(echo "$$base" | sed 's/\.example//'); \
		else \
			target=$$(echo "$$base" | sed 's/\.example$$/.env/'); \
		fi; \
		target_path="$$dir/$$target"; \
		if [ ! -f "$$target_path" ] || [ "$(FORCE)" = "yes" ]; then \
			cp "$$file" "$$target_path"; \
			echo "Copied $$file -> $$target_path"; \
		else \
			echo "$$target_path already exists, skipping"; \
		fi \
	done
endif

