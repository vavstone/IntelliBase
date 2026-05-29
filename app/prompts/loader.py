from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

def render_system_prompt(version:str = 'v1', **context)->str:
    text = (PROMPTS_DIR / f'system_{version}.md').read_text(encoding='utf-8')
    return text
    #return Template(text).render(**context)

def render_search_documents_description()->str:
    text = (PROMPTS_DIR / 'tools' / 'search_documents.md').read_text(encoding='utf-8')
    return text