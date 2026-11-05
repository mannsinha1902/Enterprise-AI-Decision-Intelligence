from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint(
            "recency_days >= 0 AND recency_days <= 180",
            name="ck_customers_recency",
        ),
        CheckConstraint("orders_90d >= 0", name="ck_customers_orders"),
        CheckConstraint("total_spend >= 0", name="ck_customers_spend"),
        CheckConstraint("support_tickets >= 0", name="ck_customers_tickets"),
        Index("ix_customers_recency_days", "recency_days"),
        Index("ix_customers_churned", "churned"),
    )

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    recency_days: Mapped[int] = mapped_column(Integer, nullable=False)
    orders_90d: Mapped[int] = mapped_column(Integer, nullable=False)
    total_spend: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    support_tickets: Mapped[int] = mapped_column(Integer, nullable=False)
    churned: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["ModelPrediction"]] = relationship(
        back_populates="customer"
    )


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("amount >= 0", name="ck_transactions_amount"),
        Index("ix_transactions_customer_id", "customer_id"),
        Index("ix_transactions_occurred_at", "occurred_at"),
    )

    transaction_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    customer: Mapped[Customer] = relationship(back_populates="transactions")


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    model_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    loaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    __table_args__ = (
        CheckConstraint(
            "churn_probability >= 0 AND churn_probability <= 1",
            name="ck_predictions_probability",
        ),
        CheckConstraint(
            "prediction IN ('churn', 'no_churn')", name="ck_predictions_label"
        ),
        Index("ix_model_predictions_customer_id", "customer_id"),
        Index("ix_model_predictions_model_version", "model_name", "model_version"),
        Index("ix_model_predictions_created_at", "created_at"),
    )

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.customer_id", ondelete="SET NULL"), nullable=True
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    recency_days: Mapped[int] = mapped_column(Integer, nullable=False)
    orders_90d: Mapped[int] = mapped_column(Integer, nullable=False)
    total_spend: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    support_tickets: Mapped[int] = mapped_column(Integer, nullable=False)
    churn_probability: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    customer: Mapped[Customer | None] = relationship(back_populates="predictions")
