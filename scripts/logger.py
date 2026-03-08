from colorama import Fore, Style
# --- LOGGER CONFIG ---
LOG_DIR = 'logs.txt'
CORE_COLOR = Fore.BLUE
APP_COLOR = Fore.YELLOW
ERROR_COLOR = Fore.RED
DEBUG_COLOR = Fore.MAGENTA
GAME_COLOR = Fore.GREEN
AR_COLOR = Fore.CYAN

COLORS = {'CORE' : CORE_COLOR, 'APP' : APP_COLOR, 'ERROR' : ERROR_COLOR, 'DEBUG' : DEBUG_COLOR, 'GAME' : GAME_COLOR, 'AR' : AR_COLOR}

def dumps(text):
    with open(LOG_DIR, 'a') as fp:
        fp.write(text)

def get_logger_info(type, text, dump=False):
    print(f"{COLORS[type]}[{type:^5}] {text}{Style.RESET_ALL}")

    if dump:
        dumps(f'\n[{type:^5}] {text}')
