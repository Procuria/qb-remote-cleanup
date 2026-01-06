# Release Notes – v0.2

This release represents the first feature-complete and publicly usable version of **qb-remote-cleanup**.
The focus of v0.2 is on safety, predictability, and operational clarity, especially for constrained
seedbox environments.

The core idea remains intentionally simple, but many rough edges have been addressed based on real-world usage.

---

## Overview

qb-remote-cleanup is a conservative cleanup tool for qBittorrent download directories that:

- Queries qBittorrent via its Web API
- Performs all filesystem operations remotely over SSH
- Identifies files that exist on disk but are no longer referenced by qBittorrent
- Supports dry-run, trash, delete, and trash-purge workflows
- Produces detailed logs and auditable summaries

---

## Features

### Orphan Detection
- Collects referenced files via torrents/info and torrents/files
- Derives scan roots automatically from save_path or via explicit configuration
- Exact-path matching
- Optional minimum-age guard
- Global and per-host path exclusions

### Cleanup Modes
- dry-run (default)
- trash
- delete

### New in v0.2: trash-purge
A dedicated mode to clean up existing trash directories.

- Deletes contents of <download_root>/<trash_subdir>
- Does not touch active download directories
- Operates purely over SSH

### New in v0.2: --trash-purge-older-than-days
Adds a safety guard to trash purging.

- Only deletes trash files older than N days
- Files with unreadable timestamps are skipped

---

## Logging and Reporting

- Structured logging with per-host context
- Per-host log files and global run log
- End-of-run summaries with file counts and sizes
- Machine-readable summary.jsonl

---

## Operational Model

- No agents or services on seedboxes
- No filesystem mounts required
- SSH-only remote execution
- Centralized administration

---

## Breaking Changes

None.

---

## Notes

This project intentionally focuses on one thing only: cleaning up qBittorrent
download directories in a way that is understandable, auditable, and hard to misuse.

Always start with dry-run.

---

## Thanks

Thanks to everyone who shared feedback and constructive discussion.
