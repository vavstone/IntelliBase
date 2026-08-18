"""add rag sources column and rag_queries table

Revision ID: f1a2b3c4d5e6
Revises: d5e6f7a8b9c0
Create Date: 2026-08-16 22:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'd5e6f7a8b9c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Показанные источники RAG-ответа рядом с assistant-сообщением.
    op.add_column(
        'chat_messages',
        sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    # Лог RAG-запросов для refusal_rate и пробелов в знаниях.
    op.create_table(
        'rag_queries',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('question_normalized', sa.String(), nullable=False),
        sa.Column('confident', sa.Boolean(), nullable=False),
        sa.Column('top_score', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('rag_queries')
    op.drop_column('chat_messages', 'sources')
