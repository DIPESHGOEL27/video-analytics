.PHONY: run test lint clean

run:
	streamlit run main.py

test:
	python -m pytest tests/ -v

lint:
	ruff check app/ tests/ main.py pages/

format:
	ruff format app/ tests/ main.py pages/

clean:
	@if exist output\*.mp4 del /Q output\*.mp4
	@if exist temp_video.mp4 del /Q temp_video.mp4
	@for /R . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
