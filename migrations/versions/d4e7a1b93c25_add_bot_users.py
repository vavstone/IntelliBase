"""add bot_users table (разрешённые пользователи бота)

Allowlist корпоративного бота: кому бот отвечает и кому агент может отправлять
сообщения. Ведётся через админ-API (`/chats/admin/bot-users`), а не правкой
`.env`: выдать доступ — админ-операция, а не деплой.

Пустая таблица = бот закрыт для всех; исключение — id из `BOT_ALLOWED_CHAT_IDS`
(bootstrap для чистого клона, где БД ещё пуста).

Revision ID: d4e7a1b93c25
Revises: c9d8e7f6a5b4
Create Date: 2026-10-05 16:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd4e7a1b93c25'
down_revision: Union[str, Sequence[str], None] = 'c9d8e7f6a5b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'bot_users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('chat_id', sa.String(length=32), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column(
            'is_active',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('true'),
        ),
        sa.Column('created_by', sa.String(length=255), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chat_id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('bot_users')
