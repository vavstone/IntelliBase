import json
import logging
from app.llm.client import LLMClient
from app.logger_setup import setup_logging
from app.tools.schemas import tools
from app.tools.handlers import search_documents


def main():
    def log_and_print(message: str) -> None:
        print(message)
        logger.info(message)

    def log(message: str) -> None:
        logger.info(message)

    print("*** Поиск документов в корпоративной базе знаний ***")
    print("Для выхода введите: /quit")
    setup_logging()
    logger = logging.getLogger(__name__)
    log_and_print("Начало работы")
    client = LLMClient(tools)
    cycle_cnt = 0
    while True:
        cycle_cnt+=1
        used_tools_count = 0
        log_and_print(f'Цикл №{cycle_cnt}')
        request = input('Вы: ')
        log(f'Пользователь: {request}')
        if request== '/quit':
            break
        answer1 = client.send({'role':'user','content':request})
        tokens = answer1[1]
        if answer1[0].content:
            log_and_print(f'Модель: {answer1[0].content}')
        if answer1[0].tool_calls:
            used_tools_count+=1
            tc = answer1[0].tool_calls[0]
            if tc and tc.function and tc.function.arguments:
                log_and_print(f'Модель: вызываю search_documents с параметрами {tc.function.arguments}')
                res = search_documents(**json.loads(tc.function.arguments))
                answer2 = client.send({'role':'tool','tool_call_id':tc.id,'content':res})
                tokens+=answer2[1]
                log_and_print(f'Модель: {answer2[0].content}')
        log_and_print(f'Задействовано tools {used_tools_count}: Использовано токенов: {tokens}')
    log_and_print("Конец работы")


if __name__ == '__main__':
    main()