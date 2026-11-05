#!/usr/bin/env bash

process_start_time() {
  local pid=$1
  ps -o lstart= -p "$pid" 2>/dev/null | xargs
}

process_matches_start_time() {
  local pid=$1
  local expected=$2
  local observed

  [[ -n "$expected" ]] || return 1
  observed=$(process_start_time "$pid")
  [[ "$observed" == "$expected" ]]
}

wait_for_process_group_leader() {
  local pid=$1
  local expected_start=$2
  local attempts=${3:-100}
  local attempt
  local pgid

  for ((attempt = 0; attempt < attempts; attempt++)); do
    process_matches_start_time "$pid" "$expected_start" || return 1
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ "$pgid" == "$pid" ]]; then
      printf '%s\n' "$pgid"
      return 0
    fi
    sleep 0.1
  done
  return 1
}

verify_source_snapshot() {
  local source_root=$1
  local manifest=$source_root/SHA256SUMS

  if [[ -s "$manifest" ]]; then
    (cd "$source_root" && sha256sum --check --quiet SHA256SUMS)
  fi
}
