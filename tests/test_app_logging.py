import logging

import modules.app_logging as app_logging


def _fresh_logger(tmp_path):
    """Points the module at a fresh temp log file and rebuilds the
    logger from scratch, since logging.getLogger() is a process-wide
    singleton keyed by name -- tests need a clean handler each time
    rather than accumulating handlers/log lines across test runs."""
    app_logging.LOG_PATH = tmp_path / "app.log"
    logging.getLogger("math_rep_system").handlers.clear()
    logger = app_logging._configure()
    app_logging.logger = logger
    return logger


def test_logger_writes_warning_to_file(tmp_path):
    logger = _fresh_logger(tmp_path)
    logger.warning("a test warning")
    content = app_logging.LOG_PATH.read_text()
    assert "a test warning" in content
    assert "WARNING" in content


def test_logger_writes_error_to_file(tmp_path):
    logger = _fresh_logger(tmp_path)
    logger.error("a test error")
    content = app_logging.LOG_PATH.read_text()
    assert "a test error" in content
    assert "ERROR" in content


def test_info_level_not_written_by_default(tmp_path):
    """The logger is configured at WARNING level -- INFO/DEBUG messages
    shouldn't appear, keeping the log to genuine failures rather than a
    full request trace."""
    logger = _fresh_logger(tmp_path)
    logger.info("should not appear")
    logger.debug("should not appear either")
    logger.warning("should appear")
    content = app_logging.LOG_PATH.read_text()
    assert "should not appear" not in content
    assert "should appear" in content


def test_reconfiguring_does_not_duplicate_handlers(tmp_path):
    """Regression test: Streamlit reruns the whole script on every
    interaction, re-executing this module's top-level code repeatedly
    in the same process. _configure() must not add a second handler
    (which would duplicate every log line) on repeated calls."""
    logger = _fresh_logger(tmp_path)
    handler_count_after_first = len(logger.handlers)
    logger2 = app_logging._configure()
    assert logger is logger2
    assert len(logger2.handlers) == handler_count_after_first


def test_log_file_created_in_expected_location(tmp_path):
    _fresh_logger(tmp_path)
    assert app_logging.LOG_PATH.parent.exists()


def test_rotating_handler_has_size_cap_and_backups(tmp_path):
    logger = _fresh_logger(tmp_path)
    handler = logger.handlers[0]
    assert handler.maxBytes == app_logging._MAX_BYTES
    assert handler.backupCount == app_logging._BACKUP_COUNT


# ---------------------------------------------------------------- gateway wiring


def test_extract_json_logs_on_failure(tmp_path):
    _fresh_logger(tmp_path)
    import modules.llm_client as llm_client
    llm_client.logger = app_logging.logger

    from modules.llm_client import extract_json, LLMOutputError
    try:
        extract_json("not valid json at all")
    except LLMOutputError:
        pass
    content = app_logging.LOG_PATH.read_text()
    assert "extract_json" in content


def test_run_with_timeout_logs_on_timeout(tmp_path):
    import time
    _fresh_logger(tmp_path)
    import modules.timeout_utils as timeout_utils
    timeout_utils.logger = app_logging.logger

    from modules.timeout_utils import run_with_timeout, ComputationTimeoutError
    try:
        run_with_timeout(lambda: time.sleep(2), timeout=0.2, label="test computation")
    except ComputationTimeoutError:
        pass
    content = app_logging.LOG_PATH.read_text()
    assert "timed out" in content.lower()
    assert "test computation" in content
