from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def print_epoch_header(stage_name: str, epoch_idx: int, total_epochs: int) -> None:
    # epoch_idx is 1-based for display
    print("============================================================")
    print(f"{stage_name} - Epoch {epoch_idx}/{total_epochs}")
    print("============================================================")


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> float:
        self.total += float(value)
        self.count += 1
        return self.value

    @property
    def value(self) -> float:
        return self.total / max(1, self.count)


def tqdm_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    leave: bool = True,
) -> Iterator[T]:
    yield from tqdm(iterable, desc=desc, dynamic_ncols=True, leave=leave)


