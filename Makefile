APP=app.py
PID_FILE=.streamlit.pid
LOG_FILE=.streamlit.log

.PHONY: build up down clean status

build:
	pip install -r requirements.txt

up:
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Streamlit already running (PID $$(cat $(PID_FILE)))"; \
		exit 0; \
	fi
	@streamlit run $(APP) --server.headless true > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)
	@echo "Streamlit started (PID $$(cat $(PID_FILE))). Logs: $(LOG_FILE)"

down:
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

clean:
	@rm -f $(PID_FILE) $(LOG_FILE)
	@rm -f outputs/workflows/* outputs/results/* 2>/dev/null || true
	@echo "Cleaned outputs and local run artifacts."

status:
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Streamlit running (PID $$(cat $(PID_FILE)))"; \
	else \
		echo "Streamlit not running"; \
	fi
