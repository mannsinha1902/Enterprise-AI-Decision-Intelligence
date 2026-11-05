"""Create synthetic customer analytics schema.

Revision ID: 0001_initial
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "customers",
        sa.Column("customer_id", sa.String(length=64), primary_key=True),
        sa.Column("recency_days", sa.Integer(), nullable=False),
        sa.Column("orders_90d", sa.Integer(), nullable=False),
        sa.Column("total_spend", sa.Numeric(12, 2), nullable=False),
        sa.Column("support_tickets", sa.Integer(), nullable=False),
        sa.Column("churned", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("recency_days >= 0 AND recency_days <= 180", name="ck_customers_recency"),
        sa.CheckConstraint("orders_90d >= 0", name="ck_customers_orders"),
        sa.CheckConstraint("total_spend >= 0", name="ck_customers_spend"),
        sa.CheckConstraint("support_tickets >= 0", name="ck_customers_tickets"),
    )
    op.create_index("ix_customers_recency_days", "customers", ["recency_days"])
    op.create_index("ix_customers_churned", "customers", ["churned"])

    op.create_table(
        "transactions",
        sa.Column("transaction_id", sa.String(length=96), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.customer_id"], ondelete="CASCADE"
        ),
        sa.CheckConstraint("amount >= 0", name="ck_transactions_amount"),
    )
    op.create_index("ix_transactions_customer_id", "transactions", ["customer_id"])
    op.create_index("ix_transactions_occurred_at", "transactions", ["occurred_at"])

    op.create_table(
        "model_metadata",
        sa.Column("model_name", sa.String(length=64), primary_key=True),
        sa.Column("model_version", sa.String(length=128), primary_key=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("loaded_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "model_predictions",
        sa.Column("prediction_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("customer_id", sa.String(length=64), nullable=True),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=True),
        sa.Column("recency_days", sa.Integer(), nullable=False),
        sa.Column("orders_90d", sa.Integer(), nullable=False),
        sa.Column("total_spend", sa.Numeric(12, 2), nullable=False),
        sa.Column("support_tickets", sa.Integer(), nullable=False),
        sa.Column("churn_probability", sa.Float(), nullable=False),
        sa.Column("prediction", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.customer_id"], ondelete="SET NULL"
        ),
        sa.CheckConstraint(
            "churn_probability >= 0 AND churn_probability <= 1",
            name="ck_predictions_probability",
        ),
        sa.CheckConstraint(
            "prediction IN ('churn', 'no_churn')", name="ck_predictions_label"
        ),
    )
    op.create_index(
        "ix_model_predictions_customer_id", "model_predictions", ["customer_id"]
    )
    op.create_index(
        "ix_model_predictions_model_version",
        "model_predictions",
        ["model_name", "model_version"],
    )
    op.create_index(
        "ix_model_predictions_created_at", "model_predictions", ["created_at"]
    )


def downgrade():
    op.drop_index("ix_model_predictions_created_at", table_name="model_predictions")
    op.drop_index(
        "ix_model_predictions_model_version", table_name="model_predictions"
    )
    op.drop_index("ix_model_predictions_customer_id", table_name="model_predictions")
    op.drop_table("model_predictions")
    op.drop_table("model_metadata")
    op.drop_index("ix_transactions_occurred_at", table_name="transactions")
    op.drop_index("ix_transactions_customer_id", table_name="transactions")
    op.drop_table("transactions")
    op.drop_index("ix_customers_churned", table_name="customers")
    op.drop_index("ix_customers_recency_days", table_name="customers")
    op.drop_table("customers")
