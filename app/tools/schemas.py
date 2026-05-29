from app.prompts.loader import render_search_documents_description


tools = [
    {
        "type":"function",
        "function":{
            "name":"search_documents",
            "description":render_search_documents_description(),
            "parameters":{
                "type":"object",
                "properties":{
                    "tags":{
                        "type":"array",
                        "items":{
                            "type":"string"
                        },
                        "description":"Список тегов (ключевых слов)"
                    },
                    "title": {"type": "string", "description": "Часть названия документа"},
                    "doc_type": {"type": "string", "description": "Тип документа (spec, db_desc, tech_req, regl, instruct, sec_policy, arch, guide, checklist, glossary, other)"},
                    "doc_number": {"type": "string", "description": "Номер документа"},
                    "doc_date_start": {"type": "string", "description": "Начало интервала для даты документа (YYYY-MM-DD)"},
                    "doc_date_end": {"type": "string", "description": "Конец интервала для даты документа (YYYY-MM-DD)"},
                    "created_at_start": {"type": "string", "description": "Начало интервала для даты добавления документа в базу знаний (YYYY-MM-DD HH:MM:SS)"},
                    "created_at_end": {"type": "string", "description": "Конец интервала для даты добавления документа в базу знаний (YYYY-MM-DD HH:MM:SS)"},
                    "updated_at_start": {"type": "string", "description": "Начало интервала для даты документа в базе знаний (YYYY-MM-DD HH:MM:SS)"},
                    "updated_at_end": {"type": "string", "description": "Конец интервала для даты документа в базе знаний (YYYY-MM-DD HH:MM:SS)"},
                },
                "additionalProperties":False
            }
        }
    }
]