"""Shared test fixtures.

The service loggers set ``propagate = False`` so container stdout does not double
up, which also means pytest's root-level caplog handler never sees their records.
Attach caplog's handler to those loggers directly so log assertions work the same
locally and on CI.
"""

import logging

import pytest

_SERVICE_LOGGERS = ("whisper_arc", "whisper_arc.translation")


@pytest.fixture(autouse=True)
def _attach_service_loggers(caplog):
    loggers = []
    for name in _SERVICE_LOGGERS:
        logger = logging.getLogger(name)
        caplog.set_level(logging.INFO, logger=name)
        logger.addHandler(caplog.handler)
        loggers.append(logger)
    yield
    for logger in loggers:
        logger.removeHandler(caplog.handler)
