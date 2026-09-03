"""Append-only journal of network changes (the write side of the stream).

`NetworkLog` hands out monotonically increasing sequence numbers and keeps
every event, including ones the projection later rejects: an attempt to close
a store that does not exist is still part of the network's history.
"""

from __future__ import annotations

from collections.abc import Iterator

from geofence.streams.schemas import STREAM_NAME, NetworkEvent


class NetworkLog:
    def __init__(self, name: str = STREAM_NAME) -> None:
        self.name = name
        self._entries: list[NetworkEvent] = []

    def append(self, event: NetworkEvent) -> NetworkEvent:
        """Stamp the next sequence number onto `event` and keep it. Returns the stamped copy."""
        stamped = event.stamped(len(self._entries) + 1)
        self._entries.append(stamped)
        return stamped

    def read(self, after_seq: int = 0) -> list[NetworkEvent]:
        """Events with seq > after_seq, oldest first."""
        if after_seq < 0:
            raise ValueError("after_seq must be >= 0")
        return self._entries[after_seq:]

    @property
    def head(self) -> int:
        return len(self._entries)

    def truncate(self) -> int:
        """Drop the whole history (used by /network/reset). Returns how many were dropped."""
        n = len(self._entries)
        self._entries.clear()
        return n

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[NetworkEvent]:
        return iter(self._entries)
