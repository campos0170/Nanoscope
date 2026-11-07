"""Plain-value session history for a figure that currently has no resident view."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .figure_spec import FrozenMap, FigureValidationError, freeze_json, thaw_json


@dataclass(frozen=True)
class HistoryCapsule:
    codec: str
    base: object
    past: tuple = ()
    future: tuple = ()
    dropped: int = 0
    extra: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self):
        if type(self.codec) is not str or not self.codec:
            raise FigureValidationError("History needs a codec identity")
        if type(self.dropped) is not int or self.dropped < 0:
            raise FigureValidationError("Dropped history count must be nonnegative")
        if not isinstance(self.past, (tuple, list)) or not isinstance(self.future, (tuple, list)):
            raise FigureValidationError("History stacks must be arrays")
        object.__setattr__(self, "base", freeze_json(self.base))
        object.__setattr__(self, "past", tuple(freeze_json(v) for v in self.past))
        object.__setattr__(self, "future", tuple(freeze_json(v) for v in self.future))
        object.__setattr__(self, "extra", FrozenMap(self.extra))

    def to_dict(self):
        return {"codec": self.codec, "base": thaw_json(self.base),
                "past": thaw_json(self.past), "future": thaw_json(self.future),
                "dropped": self.dropped, "extra": thaw_json(self.extra)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def note(self, value, *, limit=200):
        if type(limit) is not int or limit < 1:
            raise FigureValidationError("History limit must be positive")
        value = freeze_json(value)
        if value == self.base:
            return self
        past = self.past + (self.base,)
        lost = max(0, len(past) - limit)
        return replace(self, base=value, past=past[lost:], future=(), dropped=self.dropped + lost)

    def undo(self):
        if not self.past:
            return self
        return replace(self, base=self.past[-1], past=self.past[:-1], future=self.future + (self.base,))

    def redo(self):
        if not self.future:
            return self
        return replace(self, base=self.future[-1], past=self.past + (self.base,), future=self.future[:-1])
