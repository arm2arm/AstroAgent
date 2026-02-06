APP=app.py
PID_FILE=.streamlit.pid
LOG_FILE=.streamlit.log

.PHONY: help build docker-build up down logs clean fullcleanup status restart

.DEFAULT_GOAL := help

help: ## Show this help
	@echo ""
	@echo "  AstroAgent – available commands"
	@echo "  ───────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""

build: ## Install Python dependencies
	pip install -r requirements.txt

docker-build: ## Build Docker executor image (astroagent-exec)
	docker build -f Dockerfile.executor -t astroagent-exec:latest .

up: ## Start Streamlit server
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Streamlit already running (PID $$(cat $(PID_FILE)))"; \
		exit 0; \
	fi
	@streamlit run $(APP) --server.headless true > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)
	@echo "Streamlit started (PID $$(cat $(PID_FILE))). Logs: $(LOG_FILE)"

down: ## Stop Streamlit server
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID; \
			echo "Stopped Streamlit (PID $$PID)"; \
		else \
			echo "No running Streamlit process found for PID $$PID"; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		echo "No PID file found. If Streamlit is running, stop it manually."; \
	fi

logs: ## Tail the Streamlit log file
	@if [ -f $(LOG_FILE) ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "No log file found. Is Streamlit running?"; \
	fi

clean: ## Remove outputs and local artifacts
	@rm -f $(PID_FILE) $(LOG_FILE)
	@rm -f outputs/workflows/* outputs/results/* 2>/dev/null || true
	@echo "Cleaned outputs and local run artifacts."

fullcleanup: clean ## Remove outputs and local artifacts, including memory
	@rm -f outputs/memory/* 2>/dev/null || true
	@echo "Fully cleaned outputs, memory, and local run artifacts."

status: ## Show Streamlit process status
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Streamlit running (PID $$(cat $(PID_FILE)))"; \
	else \
		echo "Streamlit not running"; \
	fi

restart: down up ## Restart Streamlit server
