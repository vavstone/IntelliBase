"""Единственная реализация search_knowledge_base для обоих экспериментов.

Оба скрипта импортируют tool только отсюда — иначе сравнение single vs multi нечестное.
"""
from app.tools.graph_tools import search_knowledge_base  # реальный RAG из M5
