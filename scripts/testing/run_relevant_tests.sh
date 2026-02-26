#!/usr/bin/env bash
# Run a fast, relevant subset of tests for current changes.
#
# Modes:
#   relevant (default): infer targets from changed files (or explicit file args)
#   fast:     repo fast safety gate (non-slow, no integration)
#   full:     full suite

set -euo pipefail

MODE="relevant"
STAGED=false
LIST_ONLY=false
WITH_COV=false
RUN_GATE=false

declare -a INPUT_FILES=()
declare -a CHANGED_FILES=()
declare -a TARGETS=()

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: scripts/testing/run_relevant_tests.sh [options] [file ...]

Options:
  --relevant       Run inferred relevant tests (default)
  --fast           Run fast safety gate tests
  --full           Run full test suite
  --staged         Infer changes from staged files only (relevant mode)
  --gate           After relevant tests, also run fast safety gate
  --with-cov       Keep pytest coverage options (default is --no-cov)
  --list-only      Print selected targets/commands without running tests
  -h, --help       Show this help message

Examples:
  scripts/testing/run_relevant_tests.sh
  scripts/testing/run_relevant_tests.sh --staged --gate
  scripts/testing/run_relevant_tests.sh cohort_projections/core/migration.py
  scripts/testing/run_relevant_tests.sh --fast
  scripts/testing/run_relevant_tests.sh --full
USAGE
}

add_target_if_exists() {
  local path="$1"
  if [[ -e "$path" ]]; then
    TARGETS+=("$path")
  fi
}

infer_changed_files() {
  if [[ ${#INPUT_FILES[@]} -gt 0 ]]; then
    CHANGED_FILES=("${INPUT_FILES[@]}")
    return
  fi

  if $STAGED; then
    mapfile -t CHANGED_FILES < <(git diff --cached --name-only --diff-filter=ACMRTUXB)
  else
    mapfile -t CHANGED_FILES < <(
      {
        git diff --name-only --diff-filter=ACMRTUXB
        git ls-files --others --exclude-standard
      } | sort -u
    )
  fi
}

map_prod_file_to_tests() {
  local file="$1"
  local rel top module

  rel="${file#cohort_projections/}"
  top="${rel%%/*}"
  module="${rel##*/}"
  module="${module%.py}"

  if [[ "$module" != "__init__" ]]; then
    while IFS= read -r candidate; do
      TARGETS+=("$candidate")
    done < <(rg --files tests | rg "/test_${module}\\.py$" || true)
  fi

  case "$top" in
    core) TARGETS+=("tests/test_core") ;;
    data) TARGETS+=("tests/test_data") ;;
    geographic) TARGETS+=("tests/test_geographic") ;;
    output) TARGETS+=("tests/test_output") ;;
    utils) TARGETS+=("tests/test_utils") ;;
    *) ;;
  esac
}

infer_targets() {
  local file script_touched=false

  infer_changed_files

  for file in "${CHANGED_FILES[@]}"; do
    [[ -n "$file" ]] || continue

    case "$file" in
      tests/*.py)
        TARGETS+=("$file")
        ;;
      cohort_projections/*.py|cohort_projections/**/*.py)
        map_prod_file_to_tests "$file"
        ;;
      scripts/*)
        script_touched=true
        ;;
      pyproject.toml|.pre-commit-config.yaml)
        script_touched=true
        ;;
      *)
        ;;
    esac
  done

  # Deduplicate and keep existing paths only.
  if [[ ${#TARGETS[@]} -gt 0 ]]; then
    mapfile -t TARGETS < <(printf '%s\n' "${TARGETS[@]}" | sed '/^$/d' | sort -u)
    local filtered=()
    for file in "${TARGETS[@]}"; do
      if [[ -e "$file" ]]; then
        filtered+=("$file")
      fi
    done
    TARGETS=("${filtered[@]}")
  fi

  # If no direct targets found, use a safe fast gate fallback.
  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    if $script_touched || [[ ${#CHANGED_FILES[@]} -eq 0 ]]; then
      TARGETS=("__FAST_GATE__")
    fi
  fi
}

run_fast_gate() {
  local -a cmd=(pytest --no-cov tests/ -x -q --ignore=tests/test_integration/ -m "not slow")
  if $WITH_COV; then
    cmd=(pytest tests/ -x -q --ignore=tests/test_integration/ -m "not slow")
  fi

  echo "Running fast safety gate: ${cmd[*]}"
  if $LIST_ONLY; then
    return
  fi
  "${cmd[@]}"
}

run_full_suite() {
  local -a cmd=(pytest --no-cov tests/ -q)
  if $WITH_COV; then
    cmd=(pytest tests/ -q)
  fi

  echo "Running full suite: ${cmd[*]}"
  if $LIST_ONLY; then
    return
  fi
  "${cmd[@]}"
}

run_relevant() {
  local -a cmd

  infer_targets

  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "No Python changes mapped to tests. Nothing to run."
    return
  fi

  if [[ ${TARGETS[0]} == "__FAST_GATE__" ]]; then
    run_fast_gate
    return
  fi

  cmd=(pytest --no-cov -q)
  if $WITH_COV; then
    cmd=(pytest -q)
  fi
  cmd+=("${TARGETS[@]}")

  echo "Changed files considered: ${#CHANGED_FILES[@]}"
  echo "Relevant targets (${#TARGETS[@]}):"
  printf '  - %s\n' "${TARGETS[@]}"
  echo "Running relevant tests: ${cmd[*]}"

  if ! $LIST_ONLY; then
    "${cmd[@]}"
  fi

  if $RUN_GATE; then
    run_fast_gate
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --relevant)
      MODE="relevant"
      shift
      ;;
    --fast)
      MODE="fast"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --staged)
      STAGED=true
      shift
      ;;
    --gate)
      RUN_GATE=true
      shift
      ;;
    --with-cov)
      WITH_COV=true
      shift
      ;;
    --list-only)
      LIST_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        INPUT_FILES+=("$1")
        shift
      done
      ;;
    -* )
      die "Unknown option: $1"
      ;;
    *)
      INPUT_FILES+=("$1")
      shift
      ;;
  esac
done

case "$MODE" in
  relevant)
    run_relevant
    ;;
  fast)
    run_fast_gate
    ;;
  full)
    run_full_suite
    ;;
  *)
    die "Unsupported mode: $MODE"
    ;;
esac
