"""Timestamped progress messages shared by movie jobs and render workers."""
from datetime import datetime


def log_event(message):
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    print(f'[{stamp}] {message}', flush=True)
