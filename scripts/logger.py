from scripts.config import *

def dumps(text):
    with open(LOG_DIR, 'a') as fp:
        fp.write(text)

def get_logger_info(type, text, dump=False):
    print(f"{COLORS[type]}[{type:^5}] {text}{Style.RESET_ALL}")

    if dump:
        dumps(f'\n[{type:^5}] {text}')
