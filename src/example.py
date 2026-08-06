"""Small utilities used in early setup smoke tests."""

from __future__ import annotations


def hello_world(name: str = "World") -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


if __name__ == "__main__":
    print(hello_world())
    print(f"2 + 3 = {add_numbers(2, 3)}")
