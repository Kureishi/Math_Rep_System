"""
Basic logging: a small rotating log file (warnings and errors only, not
a full request/access log) so a recurring problem is visible after the
fact instead of only showing up as a message in the UI that's gone the
moment the page reruns. Directly motivated by an earlier incident in
this project where distinguishing "this is a rare one-off" from "this
keeps happening" for an LM Studio engine error required manually
digging back through the conversation, rather than being able to check
a log.

Deliberately narrow in scope: this logs FAILURES the app already
recovers from gracefully (an LLM call that errored, a JSON response
that didn't parse, a symbolic computation that timed out) -- not a full
request/response audit trail, which would be noisy and, for a personal
local tool with no other users to distinguish, mostly pointless
overhead.

Wired in at just three gateway points that essentially every failure of
each class already funnels through, rather than touching every
individual try/except across the app: LMStudioClient.chat() (every LLM
call, whatever the caller), extract_json() (every JSON-parsing
failure), and timeout_utils.run_with_timeout() (every symbolic
computation timeout). That gives near-complete coverage of "what's been
silently going wrong" from three edits instead of dozens.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "app.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 3             # keep up to 3 rotated-out files alongside the current one


def _configure() -> logging.Logger:
    logger = logging.getLogger("math_rep_system")
    if logger.handlers:
        # Streamlit reruns the whole script on every interaction, so this
        # module gets re-imported/re-executed repeatedly in the same
        # process -- logging.getLogger() returns the SAME logger object
        # by name each time, so this guard is what keeps a handler (and
        # therefore a log line) from being duplicated N times over.
        return logger
    logger.setLevel(logging.WARNING)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT,
                                    encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False  # don't also spam stderr/the root logger
    return logger


logger = _configure()
