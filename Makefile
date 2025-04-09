db-migrate:
	@echo "Running migrations"
	@echo "-" * 40
	python -m src.scripts.run_migrations
	@echo "-" * 40
	@echo "Migration execution finished"

db-seed:
	@echo "Inserting heart disease data"
	@echo "-" * 40
	python -m src.scripts.insert_csv_to_db
	@echo "-" * 40
	@echo "Data insertion finished"

db-setup: db-migrate db-seed


app-build:
	docker-compose build --no-cache

app-up:
	docker-compose --env-file .env up -d

run-experiment:
	@echo "Running experiment: $(F)"
	@echo "--------------------"
	python -m src.mlflow.experiments.$(F)
	@echo "--------------------"
	@echo "Experiment finished"
