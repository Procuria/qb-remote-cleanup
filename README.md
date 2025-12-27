# qBittorrent Remote Orphan Cleanup

A **safe, auditable cleanup tool** for qBittorrent download directories.

This project identifies and optionally removes **orphaned files** on remote qBittorrent hosts — files that exist on disk but are **no longer referenced by any torrent in qBittorrent**.

It is designed for:
- seedboxes
- remote hosts accessible via SSH
- setups with Sonarr / Radarr / Autobrr
- cautious operators who want **dry-runs, trash mode, logging, and summaries**

---

## ✨ Key Features

- Queries **qBittorrent Web API locally** (from the admin host)
- Performs **file operations remotely via SSH** (no script installation required on seedhosts)
- Supports **multiple qBittorrent hosts**
- **Dry-run**, **Trash**, and **Delete** modes
- Automatically derives scan roots from qBittorrent `save_path`
- Optional explicit **download root pinning**
- Per-host and global **path excludes**
- Safe handling of spaces and special characters
- Detailed logging (per host + global)
- End-of-run **human-readable summary** (counts + GB)
- JSONL machine-readable summary
- Works locally in a Python `venv`
- Docker / Coolify-ready by design

---

## 🧠 What Is an “Orphan”?

A file is considered an **orphan** if:
- it exists under a configured download root on disk
- it is **not referenced** by any torrent currently known to qBittorrent

Torrent *status* is irrelevant — only existence in qBittorrent matters.

---

## ⚠️ Safety Principles

- Dry-run by default
- Minimum file age filter
- Optional Trash mode
- Per-host and global excludes
- Full logging and summaries

---

## 📁 Project Structure

```
qb-remote-cleanup/
├── app/
│   └── main.py
├── config.yml
├── requirements.txt
├── .venv/
├── runs/
│   └── <timestamp>/
│       ├── run.log
│       ├── <host>.log
│       └── summary.jsonl
└── ssh/
    └── seedboxes
```

---

## 🔧 Requirements

**Admin Host**
- Python 3.9+
- SSH client
- Network access to qBittorrent WebUI

**Remote Hosts**
- SSH access
- bash, find, python3

---

## 🐍 Installation (Local / venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Running

```bash
python -m app.main --config ./config.yml --mode dry-run
```

Modes:
- `dry-run`
- `trash`
- `delete`

---

## 📊 Summary Output

At the end of each run, the tool prints:
- orphan count
- orphan size in GB
- per-host and global totals

A machine-readable summary is written to `summary.jsonl`.

---

## 🛡️ Recommended Workflow

1. Dry-run
2. Review logs
3. Trash mode
4. Observe
5. Delete if desired

---

## ⚠️ Disclaimer

Use at your own risk.
Always start with `dry-run`.
