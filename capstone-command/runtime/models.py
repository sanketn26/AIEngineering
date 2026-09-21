"""Task specification. Structural checks live here. Semantic holes are marked GATE."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    # extra="forbid" is the structural gate. strict=True rejects YAML enums, so it stays off.
    model_config = ConfigDict(extra="forbid")


class Priority(str, Enum):
    MUST = "must"
    SHOULD = "should"


class Command(str, Enum):
    DRAFT = "spec.draft"
    PATCH = "code.patch"
    REPAIR = "code.repair"


class Requirement(StrictModel):
    id: str = Field(pattern=r"^R[1-9][0-9]*$")
    statement: str = Field(min_length=5, max_length=500)
    priority: Priority


class AcceptanceCriterion(StrictModel):
    id: str = Field(pattern=r"^AC[1-9][0-9]*$")
    given: str = Field(min_length=3, max_length=500)
    when: str = Field(min_length=3, max_length=500)
    then: str = Field(min_length=3, max_length=500)
    # GATE 4 — planted: a free-form command string. The runtime must not shell it.
    verification: str = Field(min_length=1, max_length=500)
    maps_to: list[str] = Field(min_length=1)


class Scope(StrictModel):
    allowed_files: list[str] = Field(min_length=1, max_length=5)
    max_files_changed: int = Field(ge=1, le=5)
    max_added_lines: int = Field(ge=1, le=500)

    @field_validator("allowed_files")
    @classmethod
    def safe_paths(cls, files: list[str]) -> list[str]:
        for raw in files:
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts or raw.startswith("/"):
                raise ValueError(f"unsafe path: {raw}")
        return files


class Constraints(StrictModel):
    allow_new_dependencies: bool = False
    network_access: bool = False
    preserve_public_api: bool = True


class TaskSpec(StrictModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    task_id: str = Field(min_length=1, max_length=80)
    command: Command
    goal: str = Field(min_length=5, max_length=500)
    scope: Scope
    requirements: list[Requirement] = Field(min_length=1)
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1)
    non_goals: list[str] = Field(min_length=1)
    unknowns: list[str] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    # GATE 4 — planted: a label carried on the spec. eligible_for_4b trusts it.
    risk_level: str = "low"

    @model_validator(mode="after")
    def graph_is_well_formed(self):
        requirement_ids = [item.id for item in self.requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("duplicate requirement ids")
        criterion_ids = [item.id for item in self.acceptance_criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("duplicate acceptance criterion ids")
        known = set(requirement_ids)
        referenced = {
            requirement_id
            for criterion in self.acceptance_criteria
            for requirement_id in criterion.maps_to
        }
        unknown_refs = referenced - known
        if unknown_refs:
            raise ValueError(f"unknown requirement references: {sorted(unknown_refs)}")
        # GATE 2 — planted: a must requirement with no criterion still validates.
        # GATE 1 — planted: a non-empty unknowns list still validates.
        return self
