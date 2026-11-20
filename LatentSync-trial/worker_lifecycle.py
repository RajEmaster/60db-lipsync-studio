"""Stop render descendants when their owning UI exits."""
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil


@contextmanager
def exclusive_job_resource(filename, waiting_message):
    """Serialize access across app processes; the OS releases the lock on exit."""
    from locking_compat import try_lock, unlock
    folder = Path(__file__).resolve().parent / 'local_jobs'
    folder.mkdir(exist_ok=True)
    with (folder / filename).open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        waiting = False
        while not try_lock(handle):
            if not waiting:
                print(waiting_message, flush=True)
                waiting = True
            time.sleep(1)
        try:
            yield
        finally:
            unlock(handle)


def exclusive_gpu():
    return exclusive_job_resource('gpu.lock', 'Another render owns the GPU; waiting for it to finish.')


def watch_parent(pid):
    try:
        parent = psutil.Process(pid)
        started = parent.create_time()
    except psutil.NoSuchProcess:
        raise RuntimeError('The owning UI has already exited.') from None

    def monitor():
        while True:
            threading.Event().wait(2)
            try:
                if parent.is_running() and parent.create_time() == started:
                    continue
            except psutil.NoSuchProcess:
                pass
            print('Owning UI exited; stopping render workers and releasing GPU memory.', flush=True)
            children = psutil.Process().children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            os._exit(1)

    threading.Thread(target=monitor, daemon=True).start()
