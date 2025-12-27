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
- Performs **file operations remotely via SSH** (no scripts required on seedhosts)
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
- Docker / docker-compose friendly

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

## 🔧 Configuration Reference

### 🔑 SSH Requirements

- Key-based authentication only
- SSH user must have read/write permissions
- No sudo required
- SSH keys must not be committed to the repository

---

### Global Options

```yaml
mode: dry-run                # dry-run | trash | delete
min_age_minutes: 240         # safety window before touching files
clean_empty_dirs: true

trash_subdir: ".trash/qb_orphans"
trash_retention_days: 14

out_dir: "./runs"

exclude_paths_containing:
  - "/.trash/"
```

### Per-Host Options

```yaml
hosts:
  - name: "SeedHost Bee"

    qb:
      url: "https://seedbox.example.com/qbittorrent/"
      username: "user"
      password: "secret"
      verify_tls: true

    ssh:
      host: "seedbox.example.com"
      user: "user"
      key_path: "./ssh/seedboxes"

    auto_roots: false
    download_roots:
      - "/downloads"

    exclude_paths_containing:
      - "/opt/docker_volumes/"
```

---

## 🐳 Docker-Compose Example

```yaml
version: "3.9"
services:
  qb-remote-cleanup:
    image: python:3.11-slim
    volumes:
      - ./app:/app/app
      - ./config.yml:/app/config.yml:ro
      - ./ssh:/app/ssh:ro
      - ./runs:/app/runs
    working_dir: /app
    command: >
      sh -c "pip install -r app/requirements.txt &&
             python -m app.main --config /app/config.yml --mode dry-run"
```

---

## 🛠️ Troubleshooting

- **Unexpected files**: Disable `auto_roots` and pin `download_roots`
- **Permission errors**: Verify SSH user ownership
- **Too many `.parts` files**: Increase `min_age_minutes`

---

## 🔐 Security Notes

- Do not commit credentials
- Use separate configs per environment
- Use least-privilege SSH users

---

## ⚠️ Disclaimer

Always start with `dry-run`.
