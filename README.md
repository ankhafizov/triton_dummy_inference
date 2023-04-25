# Простой проект с технологией triton inference server

Основа: https://github.com/EgShes/one_task_multiple_infras

Статья на хабре: https://habr.com/ru/articles/717890/

Запуск:
0. Установить nvidia-docker и зависимости из requirements 
1. Запуск сервера: `docker-compose up`
2. Запуск клиентов: `python client_lpr_photo.py` или `python client_lpr_video.py`
