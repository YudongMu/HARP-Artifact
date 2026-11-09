"""A scheduled operation on a component resource."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    resource: str
    cycles: int
    after: tuple = ()
