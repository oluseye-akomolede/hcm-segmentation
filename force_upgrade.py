from alembic.config import Config
from alembic import command
import os

base_path = os.path.abspath(".venv/lib/python3.12/site-packages/mlflow/store/db_migrations")

# Use any real file for logging config — we're bypassing logging anyway
ini_path = os.path.join(base_path, "alembic.ini")

alembic_cfg = Config(ini_path)
alembic_cfg.set_main_option("script_location", base_path)
alembic_cfg.set_main_option("sqlalchemy.url", os.environ["MLFLOW_DB_URI"])

# Run the upgrade
command.upgrade(alembic_cfg, "7f2a7d5fae7d")

