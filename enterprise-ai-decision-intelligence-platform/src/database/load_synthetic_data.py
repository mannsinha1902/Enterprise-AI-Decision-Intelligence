import os

from alembic import command
from alembic.config import Config

from src.database.operations import load_synthetic_customers
from src.database.session import create_database_engine, create_session_factory


def main():
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError(
            "Set DATABASE_URL to a local database connection string."
        )

    config = Config("alembic.ini")
    command.upgrade(config, "head")
    engine = create_database_engine()
    try:
        customer_count, transaction_count = load_synthetic_customers(
            create_session_factory(engine)
        )
    finally:
        engine.dispose()
    print(
        f"Loaded {customer_count} synthetic customers and "
        f"{transaction_count} deterministic synthetic transactions."
    )


if __name__ == "__main__":
    main()
