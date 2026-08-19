"""add kb_categories table + seed

Категории знаний = названия ПС (подсистем ФТС). Ключ — латинский slug
(безопасен в путях/URL/callback), отображение — русский title. Seed
идемпотентен (ON CONFLICT DO NOTHING) — повторный прогон миграции не дублирует.

Revision ID: c9d8e7f6a5b4
Revises: f1a2b3c4d5e6
Create Date: 2026-08-18 15:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c9d8e7f6a5b4'
down_revision: Union[str, Sequence[str], None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Таксономия ПС (см. docs/tech_debt/category-taxonomy-ps-*.md, п.2).
_CATEGORY_SEED: list[tuple[str, str]] = [
    ("tarify", "Тарифы"),
    ("malahit", "Малахит"),
    ("postkontrol", "Постконтроль"),
    ("tamozhnya_pravo", "Таможня и право"),
    ("pravoohrana", "Правоохрана"),
    ("crsved", "ЦРСВЭД"),
    ("raznoe", "Разное"),
]


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'kb_categories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('slug', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slug'),
    )

    # Data-migration: seed каталога категорий. Идемпотентно по slug.
    rows = ", ".join(
        f"('{slug}', '{title}')" for slug, title in _CATEGORY_SEED
    )
    op.execute(
        f"""
        INSERT INTO kb_categories (slug, title) VALUES {rows}
        ON CONFLICT (slug) DO NOTHING
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('kb_categories')
