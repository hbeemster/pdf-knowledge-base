set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# list all tasks
default:
  @just --list

# format code using ruff
format:
    uv run ruff format

# check code using ruff
check:
    uv run ruff check --fix

# start optuna dashboard
dashboard:
    wt -- uv run optuna-dashboard ./data/optuna/journal.log
