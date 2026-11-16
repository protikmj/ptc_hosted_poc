# UV Dependency Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Manage local and Databricks notebook dependencies through `pyproject.toml` and `uv.lock`, with no `requirements.txt`.

**Architecture:** Declare existing dependencies in the project metadata and let `uv` generate the exact lock. The notebook uses Databricks Shared Serverless environment 5+ and `%uv sync`; its existing project-root `sys.path` setup continues to load `ptc_demo`.

**Tech Stack:** Python 3.10+, uv, Databricks Shared Serverless environment 5+.

**Spec:** `docs/superpowers/specs/2026-11-16-uv-dependency-management-design.md`

## Global Constraints

- Keep Python 3.10+ support.
- Preserve dependency ranges: `openai==3.50.0`, `matplotlib>=3.9,<4`, `plotly>=6,<7`.
- Use Databricks Shared Serverless environment version `5` and `%uv sync`.
- Do not change application behavior or add dependencies.

## Review Focus

- The dependency specifiers must match the existing `requirements.txt` entries exactly.
- The lockfile must be generated from the manifest, not edited by hand.
- The notebook must retain its project-root import setup after dependency installation.
- Setup docs must contain no remaining `requirements.txt` instructions.
- Classic compute and serverless environments below version 5 are outside this setup's scope.

---

### Task 1: Move project and notebook setup to uv

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock` with uv
- Delete: `requirements.txt`
- Modify: `ptc_demo/runners.py` SDK compatibility message
- Modify: `notebooks/ptc_demo.py` setup instructions and setup cell
- Modify: `README.md` local commands and Databricks instructions

- [ ] Add `[project]` metadata to `pyproject.toml` with `requires-python = ">=3.10"` and the three existing dependency specifiers. Add `[tool.databricks.environment]` with `environment_version = "5"`.
- [ ] Run `uv lock` to resolve and create the checked-in `uv.lock`.
- [ ] Remove `requirements.txt`.
- [ ] Point the SDK compatibility error in `ptc_demo/runners.py` to `uv sync` locally and `%uv sync` in Databricks Shared Serverless environment 5+.
- [ ] Update README setup and command examples to use `uv sync` and `uv run`; replace Databricks installation steps with Shared Serverless environment 5+ and `%uv sync`.
- [ ] Update the notebook setup cell to run `%uv sync` and explain its Shared Serverless environment 5+ requirement. Keep the existing `PROJECT_ROOT` and `sys.path` behavior.
- [ ] Review the diff for dependency parity and remove all `requirements.txt` references.
