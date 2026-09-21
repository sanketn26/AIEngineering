"""User task, user checks, and the part records a divider is allowed to propose."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Requirement(StrictModel):
    id: str = Field(pattern=r"^R[1-9][0-9]*$")
    statement: str = Field(min_length=5, max_length=500)
    priority: Literal["must", "should"]


class Check(StrictModel):
    id: str = Field(pattern=r"^V[1-9][0-9]*$")
    scope: Literal["part", "join"]
    maps_to: list[str] = Field(min_length=1)
    runner: str = Field(min_length=1, max_length=40)
    target: str = Field(min_length=1, max_length=200)


class Scope(StrictModel):
    allowed_files: list[str] = Field(min_length=1, max_length=12)
    max_files_changed: int = Field(ge=1, le=12)
    max_added_lines: int = Field(ge=1, le=2000)

    @field_validator("allowed_files")
    @classmethod
    def safe_paths(cls, files: list[str]) -> list[str]:
        for raw in files:
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe path: {raw}")
        return files


class Seam(StrictModel):
    name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")
    kind: Literal["function", "type", "fixture"]
    signature: str = Field(min_length=1, max_length=200)


class Part(StrictModel):
    id: str = Field(pattern=r"^P[1-9][0-9]*$")
    owns: list[str] = Field(min_length=1)
    # No pattern here on purpose. Gate 2 must reject ids the user did not write.
    checks: list[str] = Field(min_length=1)
    files: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    exports: list[Seam] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)


class TaskSpec(StrictModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    task_id: str = Field(min_length=1, max_length=80)
    goal: str = Field(min_length=5, max_length=500)
    requirements: list[Requirement] = Field(min_length=1)
    verification: list[Check] = Field(min_length=1)
    scope: Scope
    unknowns: list[str] = Field(default_factory=list)


class Plan(StrictModel):
    parts: list[Part] = Field(min_length=1)
