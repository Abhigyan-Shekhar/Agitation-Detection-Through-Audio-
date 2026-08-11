"""Bounded non-blocking queue fan-out helpers."""
from __future__ import annotations

import logging
import queue
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def as_queue_list(targets: queue.Queue[T] | Iterable[queue.Queue[T]]) -> list[queue.Queue[T]]:
    return [targets] if isinstance(targets, queue.Queue) else list(targets)


def publish_latest(
    targets: Iterable[queue.Queue[T]],
    item: T,
    *,
    logger: logging.Logger,
    label: str,
) -> int:
    """Publish independently to every bounded consumer queue, dropping only its oldest item."""
    delivered = 0
    for target in targets:
        try:
            target.put_nowait(item)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(item)
            except queue.Full:
                logger.warning("%s full — item dropped for one consumer", label)
                continue
        delivered += 1
    return delivered
