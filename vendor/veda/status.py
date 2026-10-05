"""VEDA runtime diagnostics are written only to the console log."""

from __future__ import annotations

import logging

_LOG = logging.getLogger('veda')


class NodeStatus:
    """Logs runtime status without adding a node text widget."""

    def __init__(self, node_id: str | None):
        del node_id

    def show(self, text: str, level: int = logging.INFO) -> None:
        # Status text carries no emoji, so the only non-ASCII left is the
        # ' · ' separator; the log still gets plain ASCII, because Windows
        # consoles and log files may not be UTF-8. The strip stays as the
        # backstop that keeps that promise whatever a caller passes in.
        plain = text.replace(' · ', ' | ').encode('ascii', 'ignore')
        plain = plain.decode().strip()
        _LOG.log(level, 'Veda: %s', plain)

    def warn(self, text: str) -> None:
        self.show(text, logging.WARNING)
