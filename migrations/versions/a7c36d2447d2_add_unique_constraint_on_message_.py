"""add unique constraint on message_feedback

Revision ID: a7c36d2447d2
Revises: e19607a58f0d
Create Date: 2026-07-12 17:18:59.451772

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7c36d2447d2'
down_revision: Union[str, Sequence[str], None] = 'e19607a58f0d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_unique_constraint(
        'uq_message_feedback_owner_msg',
        'message_feedback',
        ['owner_external_id', 'message_id'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        'uq_message_feedback_owner_msg',
        'message_feedback',
        type_='unique',
    )
