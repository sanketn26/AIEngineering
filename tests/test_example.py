"""Tests for the example module."""

from src.example import add_numbers, hello_world


def test_hello_world_default():
    assert hello_world() == "Hello, World!"


def test_hello_world_custom_name():
    assert hello_world("Alice") == "Hello, Alice!"


def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0


def test_add_numbers_negative():
    assert add_numbers(-5, -3) == -8
    assert add_numbers(-10, 5) == -5
