"""
SQLite connection handling shared by the app's four small databases
(history, problem chains, templates, settings profiles).

`with sqlite3.connect(path) as conn:` does NOT close the connection: the
context manager only commits (or rolls back) the transaction and leaves the
connection open until it is garbage-collected. That was harmless for years,
but Python 3.13 started emitting a ResourceWarning for every connection
dropped unclosed (hundreds per test run on 3.14), and an open connection also
keeps a WAL-mode database's -wal and -shm files locked -- which on Windows
stops the folder being cleaned up or the file replaced.

ClosingConnection keeps the `with _connect() as conn:` call sites exactly as
they were, and makes leaving the block do what the code always assumed it
did: commit (or roll back on an exception), then close.
"""
import sqlite3


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)   # commit, or roll back if it raised
        finally:
            self.close()
