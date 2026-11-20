"""Cross-platform advisory file locking.

Windows uses msvcrt byte-range locks; POSIX uses fcntl.flock. Both give the
cooperating app processes a simple cross-process mutex over a small lock file,
released automatically when the owning handle is closed or the process exits.
"""
import sys

if sys.platform == 'win32':
    import msvcrt

    def try_lock(handle):
        """Non-blocking exclusive lock on one byte. True if acquired, False if held."""
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def unlock(handle):
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def try_lock(handle):
        """Non-blocking exclusive lock. True if acquired, False if another holder has it."""
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def unlock(handle):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
