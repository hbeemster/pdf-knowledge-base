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

# run RAG optimizer: just optimize <study-name> [n-trials] [n-processes]
optimize study_name n_trials="3" n_processes="":
    uv run optimize-rag {{study_name}} --n-trials {{n_trials}} {{ if n_processes != "" { "--n-processes " + n_processes } else { "" } }}

# start optuna dashboard
dashboard:
    wt -- uv run optuna-dashboard ./data/optuna/journal.log
