eval "$(conda shell.bash hook)"
conda activate gid
python -m db.init_db
python -m db.load_fixtures
uvicorn main:app --reload

