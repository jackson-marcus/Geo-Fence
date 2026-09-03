"""Event contracts for the store-network change log.

The network a gravity model scores is not static: outlets open, close and get
refitted. Each of those is a `NetworkEvent`. The log is the source of truth;
the store table the API serves is a projection folded from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

STREAM_NAME = "network.changes"
EventKind = Literal["open", "close", "resize"]
EVENT_KINDS: tuple[str, ...] = ("open", "close", "resize")
MAX_STORE_SQM = 20000.0


class InvalidEventError(ValueError):
    """The event is malformed on its own terms (before looking at network state)."""


@dataclass(frozen=True)
class NetworkEvent:
    kind: str
    store_id: int
    x: float | None = None
    y: float | None = None
    size_sqm: float | None = None
    note: str = ""
    seq: int = 0  # assigned by the log on append; 0 = not yet appended
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise InvalidEventError(
                f"unknown event kind {self.kind!r}; expected one of {EVENT_KINDS}"
            )
        if int(self.store_id) != self.store_id or self.store_id <= 0:
            raise InvalidEventError("store_id must be a positive integer")
        if self.kind == "open":
            if self.x is None or self.y is None:
                raise InvalidEventError("open needs x and y")
            if self.x < 0 or self.y < 0:
                raise InvalidEventError("coordinates must be non-negative grid positions")
        if self.kind in ("open", "resize"):
            if self.size_sqm is None:
                raise InvalidEventError(f"{self.kind} needs size_sqm")
            if not (0 < self.size_sqm <= MAX_STORE_SQM):
                raise InvalidEventError(f"size_sqm must be in (0, {MAX_STORE_SQM:.0f}]")
        if self.kind == "close" and any(v is not None for v in (self.x, self.y, self.size_sqm)):
            raise InvalidEventError("close takes only a store_id")

    @classmethod
    def open(
        cls, store_id: int, x: float, y: float, size_sqm: float, note: str = ""
    ) -> NetworkEvent:
        return cls("open", store_id, float(x), float(y), float(size_sqm), note)

    @classmethod
    def close(cls, store_id: int, note: str = "") -> NetworkEvent:
        return cls("close", store_id, note=note)

    @classmethod
    def resize(cls, store_id: int, size_sqm: float, note: str = "") -> NetworkEvent:
        return cls("resize", store_id, size_sqm=float(size_sqm), note=note)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> NetworkEvent:
        """Build from an API body; unknown keys are ignored, missing ones validated."""
        allowed = {"kind", "store_id", "x", "y", "size_sqm", "note"}
        try:
            return cls(**{k: v for k, v in payload.items() if k in allowed})
        except TypeError as exc:  # missing positional args
            raise InvalidEventError(str(exc)) from exc

    def stamped(self, seq: int) -> NetworkEvent:
        return replace(self, seq=seq)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "store_id": self.store_id,
            "x": self.x,
            "y": self.y,
            "size_sqm": self.size_sqm,
            "note": self.note,
        }
