setup:
	pip install -r requirements.txt

ingest:
	python src/ingestion.py

ui:
	streamlit run src/chat_ui.py --server.headless true