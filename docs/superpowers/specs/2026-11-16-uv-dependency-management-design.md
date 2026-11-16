# UV managed dependencies

## Intent

Manage the project's Python dependencies with `uv` from one `pyproject.toml` and a checked-in `uv.lock`, including the Databricks notebook setup.

## Goals and constraints

- Remove `requirements.txt` and update all setup instructions that refer to it.
- Keep the existing dependency ranges and Python 3.10+ support.
- Use Databricks Shared Serverless environment 5+ and its `%uv sync` notebook command.
- Do not add dependencies or change application behavior.

## Non-goals

- Support classic Databricks compute in this setup.
- Change dependency versions or add packaging/CLI features.

## Approaches

1. Use `pyproject.toml` plus `uv.lock` locally and `%uv sync` in Databricks Shared Serverless environment 5+. **Selected** because this is the user's target and lets local and notebook environments use the same lock.
2. Use `uv` locally, then `%pip install -e <project path>` in Databricks. This works on more compute types, but the notebook install resolves from package metadata instead of applying `uv.lock`.

## Implementation

Put the current dependencies and Python range in `pyproject.toml`, declare Databricks environment version 5, generate and retain `uv.lock`, remove `requirements.txt`, and change the notebook and README to use `uv sync`/`uv run` and `%uv sync`. The notebook already adds the project root to `sys.path`, so this change does not require a new package build configuration.

## Remaining implementation call

Use the current dependency specifiers as-is; let `uv.lock` select exact versions for reproducible installs.
