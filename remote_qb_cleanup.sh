#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# qb-remote-cleanup runner (interactive)
# -----------------------------------------------------------------------------
# Goals:
# - Make running the tool dead simple from shell
# - Encourage safe workflows (dry-run -> trash -> optional delete)
# - Minimize typing and prevent "oops" mode runs
# -----------------------------------------------------------------------------

DEFAULT_CONFIG="./config.yml"
DEFAULT_LOG_LEVEL="INFO"
DEFAULT_OUT_DIR=""   # empty = use config value
DEFAULT_MIN_AGE=""   # empty = use config value
DEFAULT_PURGE_OLDER="" # empty = purge all trash

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODULE="app.main"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
warn() { printf "\033[33m%s\033[0m\n" "$*"; }
err()  { printf "\033[31m%s\033[0m\n" "$*"; }
ok()   { printf "\033[32m%s\033[0m\n" "$*"; }

need() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing dependency: $1"; exit 1; }
}

need "$PYTHON_BIN"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# Try to locate venv python if present
if [[ -x "$repo_root/.venv/bin/python" ]]; then
  PYTHON_BIN="$repo_root/.venv/bin/python"
fi

# Helper: prompt with default
prompt() {
  local label="$1"
  local default="${2:-}"
  local input
  if [[ -n "$default" ]]; then
    read -r -p "$label [$default]: " input
    echo "${input:-$default}"
  else
    read -r -p "$label: " input
    echo "$input"
  fi
}

# Helper: yes/no
confirm() {
  local label="$1"
  local input
  read -r -p "$label (yes/no): " input
  case "${input,,}" in
    yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

# Find configs (nice-to-have)
configs=()
if compgen -G "./config*.yml" >/dev/null; then
  while IFS= read -r f; do configs+=("$f"); done < <(ls -1 ./config*.yml 2>/dev/null || true)
fi
if [[ -f "$DEFAULT_CONFIG" && "${#configs[@]}" -eq 0 ]]; then
  configs+=("$DEFAULT_CONFIG")
fi

bold "qb-remote-cleanup runner"
echo "Repo: $repo_root"
echo "Python: $PYTHON_BIN"
echo

# Select config
CONFIG="$DEFAULT_CONFIG"
if [[ "${#configs[@]}" -gt 0 ]]; then
  echo "Select config:"
  i=1
  for c in "${configs[@]}"; do
    echo "  $i) $c"
    ((i++))
  done
  echo "  $i) Enter path manually"
  choice="$(prompt "Choice" "1")"
  if [[ "$choice" =~ ^[0-9]+$ ]]; then
    if (( choice >= 1 && choice <= ${#configs[@]} )); then
      CONFIG="${configs[$((choice-1))]}"
    elif (( choice == ${#configs[@]} + 1 )); then
      CONFIG="$(prompt "Config path" "$DEFAULT_CONFIG")"
    else
      warn "Invalid choice, using default: $DEFAULT_CONFIG"
      CONFIG="$DEFAULT_CONFIG"
    fi
  else
    warn "Invalid input, using default: $DEFAULT_CONFIG"
    CONFIG="$DEFAULT_CONFIG"
  fi
else
  CONFIG="$(prompt "Config path" "$DEFAULT_CONFIG")"
fi

if [[ ! -f "$CONFIG" ]]; then
  err "Config file not found: $CONFIG"
  exit 1
fi

echo
echo "Select mode:"
echo "  1) dry-run     (safe, default)"
echo "  2) trash       (moves orphans into trash)"
echo "  3) delete      (permanently deletes orphans)"
echo "  4) trash-purge (purges existing trash directories)"
mode_choice="$(prompt "Choice" "1")"

MODE="dry-run"
case "$mode_choice" in
  1) MODE="dry-run" ;;
  2) MODE="trash" ;;
  3) MODE="delete" ;;
  4) MODE="trash-purge" ;;
  *) warn "Unknown choice; using dry-run"; MODE="dry-run" ;;
esac

echo
LOG_LEVEL="$(prompt "Log level (DEBUG/INFO/WARNING/ERROR)" "$DEFAULT_LOG_LEVEL")"

# Optional overrides
echo
MIN_AGE="$(prompt "Override min_age_minutes (empty = use config)" "$DEFAULT_MIN_AGE")"
OUT_DIR="$(prompt "Override out_dir (empty = use config)" "$DEFAULT_OUT_DIR")"

PURGE_OLDER=""
if [[ "$MODE" == "trash-purge" ]]; then
  echo
  PURGE_OLDER="$(prompt "trash-purge older-than-days (empty = purge all)" "$DEFAULT_PURGE_OLDER")"
fi

# Safety confirmations for destructive modes
if [[ "$MODE" == "delete" ]]; then
  warn ""
  warn "You selected DELETE mode. This will permanently remove orphaned files."
  warn "Recommendation: run dry-run first, then trash, then delete."
  if ! confirm "Proceed with DELETE mode?"; then
    ok "Aborted."
    exit 0
  fi
fi

if [[ "$MODE" == "trash-purge" ]]; then
  warn ""
  warn "You selected TRASH-PURGE mode. This deletes contents under <root>/<trash_subdir>."
  if [[ -n "$PURGE_OLDER" ]]; then
    warn "It will only delete trash files older than $PURGE_OLDER day(s)."
  else
    warn "It will delete ALL trash files (no age filter)."
  fi
  if ! confirm "Proceed with TRASH-PURGE?"; then
    ok "Aborted."
    exit 0
  fi
fi

# Build command
cmd=( "$PYTHON_BIN" -m "$MODULE" --config "$CONFIG" --mode "$MODE" --log-level "$LOG_LEVEL" )

if [[ -n "$MIN_AGE" ]]; then
  cmd+=( --min-age-minutes "$MIN_AGE" )
fi

if [[ -n "$OUT_DIR" ]]; then
  cmd+=( --out-dir "$OUT_DIR" )
fi

if [[ "$MODE" == "trash-purge" && -n "$PURGE_OLDER" ]]; then
  cmd+=( --trash-purge-older-than-days "$PURGE_OLDER" )
fi

echo
bold "Running:"
printf "  %q" "${cmd[@]}"
echo
echo

# Execute
"${cmd[@]}"

echo
ok "Done."

# Nice-to-have: show latest run directory if using default ./runs
# This is best-effort and won’t fail the script.
if [[ -z "$OUT_DIR" ]]; then
  # out_dir comes from config, but many setups use ./runs
  if [[ -d "./runs" ]]; then
    last="$(ls -1dt ./runs/* 2>/dev/null | head -n 1 || true)"
    if [[ -n "$last" ]]; then
      echo "Latest run output: $last"
    fi
  fi
else
  if [[ -d "$OUT_DIR" ]]; then
    last="$(ls -1dt "$OUT_DIR"/* 2>/dev/null | head -n 1 || true)"
    if [[ -n "$last" ]]; then
      echo "Latest run output: $last"
    fi
  fi
fi
