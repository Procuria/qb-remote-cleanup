#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
import yaml


# -----------------------------
# Logging
# -----------------------------
class HostAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = dict(kwargs.get("extra", {}) or {})
        # prevent overwrite of "host" if someone passes it in extra
        extra.pop("host", None)
        merged = {**self.extra, **extra}
        kwargs["extra"] = merged
        return msg, kwargs


class EnsureHostFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "host"):
            record.host = "-"
        return True


def setup_logging(out_dir: Path, level: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("qb_remote_cleanup")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.filters.clear()

    fmt = "%(asctime)s %(levelname)s host=%(host)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"

    logger.addFilter(EnsureHostFilter())

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(ch)

    # Master log file
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(fh)

    logger.info("Logging initialized", extra={"host": "-"})
    return logger


# -----------------------------
# Helpers
# -----------------------------
GB = 1000 ** 3


def bytes_to_gib(b: int) -> float:
    return b / GB


# -----------------------------
# Config models
# -----------------------------
@dataclass
class SSHConfig:
    host: str
    user: str
    port: int = 22
    key_path: Optional[str] = None
    strict_host_key_checking: str = "accept-new"  # yes|no|accept-new
    known_hosts_file: Optional[str] = None


@dataclass
class QBConfig:
    url: str
    username: str
    password: str
    verify_tls: bool = True
    timeout_seconds: int = 15


@dataclass
class HostConfig:
    name: str
    qb: QBConfig
    ssh: SSHConfig
    # If not set and auto_roots=true => derive from qb save_path
    download_roots: Optional[List[str]] = None
    auto_roots: bool = True
    # Per-host excludes (strings that must NOT appear in candidate paths)
    exclude_paths_containing: Optional[List[str]] = None


@dataclass
class AppConfig:
    hosts: List[HostConfig]
    min_age_minutes: int = 240
    mode: str = "dry-run"  # dry-run | trash | delete
    clean_empty_dirs: bool = True
    trash_subdir: str = ".trash/qb_orphans"
    trash_retention_days: int = 14
    out_dir: str = "./runs"
    # Safety
    exclude_paths_containing: List[str] = None  # global excludes


# -----------------------------
# qBittorrent API (queried locally)
# -----------------------------
class QBClient:
    def __init__(self, cfg: QBConfig):
        self.cfg = cfg
        self.sess = requests.Session()
        self.sess.verify = cfg.verify_tls

    def _url(self, path: str) -> str:
        return self.cfg.url.rstrip("/") + path

    def login(self) -> None:
        r = self.sess.post(
            self._url("/api/v2/auth/login"),
            data={"username": self.cfg.username, "password": self.cfg.password},
            timeout=self.cfg.timeout_seconds,
        )
        if r.status_code != 200 or r.text.strip() != "Ok.":
            raise RuntimeError(
                f"qBittorrent login failed: status={r.status_code} body={r.text[:200]}"
            )

    def torrents_info(self) -> List[dict]:
        r = self.sess.get(self._url("/api/v2/torrents/info"), timeout=self.cfg.timeout_seconds)
        r.raise_for_status()
        return r.json()

    def torrent_files(self, torrent_hash: str) -> List[dict]:
        r = self.sess.get(
            self._url("/api/v2/torrents/files"),
            params={"hash": torrent_hash},
            timeout=self.cfg.timeout_seconds,
        )
        r.raise_for_status()
        return r.json()


# -----------------------------
# SSH helpers (only deletion/listing remote)
# -----------------------------
def build_ssh_base_cmd(ssh: SSHConfig) -> List[str]:
    cmd = [
        "ssh",
        "-p",
        str(ssh.port),
        "-o",
        "BatchMode=yes",
        "-o",
        f"StrictHostKeyChecking={ssh.strict_host_key_checking}",
    ]
    if ssh.known_hosts_file:
        cmd += ["-o", f"UserKnownHostsFile={ssh.known_hosts_file}"]
    if ssh.key_path:
        cmd += ["-i", ssh.key_path]
    cmd.append(f"{ssh.user}@{ssh.host}")
    return cmd


def ssh_run(
    ssh: SSHConfig,
    stdin_bytes: bytes,
    timeout: int = 600,
) -> Tuple[int, bytes, bytes]:
    """
    Runs 'bash -s' on remote host, feeding stdin_bytes as the script contents.
    """
    cmd = build_ssh_base_cmd(ssh) + ["bash", "-s"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = p.communicate(input=stdin_bytes, timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, (err + b"\nTIMEOUT\n")


# -----------------------------
# Core logic
# -----------------------------
def normalize_path(p: str) -> str:
    # Keep Linux-style; do minimal normalization
    return p.replace("//", "/")


def derive_scan_roots(host: HostConfig, torrents: List[dict]) -> List[str]:
    if host.download_roots:
        return [normalize_path(r.rstrip("/")) for r in host.download_roots]
    if host.auto_roots:
        roots = sorted(
            {
                normalize_path((t.get("save_path") or "").rstrip("/"))
                for t in torrents
                if (t.get("save_path") or "").strip()
            }
        )
        return [r for r in roots if r]
    raise RuntimeError(f"{host.name}: No download_roots set and auto_roots=false")


def build_referenced_paths(host_log: HostAdapter, qb: QBClient, torrents: List[dict]) -> Set[str]:
    """
    Build a set of absolute file paths: save_path + relative file name from torrents/files.
    """
    ref: Set[str] = set()

    # map hash -> save_path (cached from torrents/info)
    save_map: Dict[str, str] = {}
    for t in torrents:
        h = t.get("hash")
        sp = t.get("save_path") or ""
        if h and sp.strip():
            save_map[h] = normalize_path(sp.rstrip("/"))

    hashes = list(save_map.keys())
    host_log.info(f"Found torrents with save_path: {len(hashes)} (from torrents/info)")

    for i, h in enumerate(hashes, start=1):
        files = qb.torrent_files(h)
        sp = save_map[h]
        for f in files:
            rel = f.get("name") or ""
            if not rel:
                continue
            abs_path = normalize_path(f"{sp}/{rel}")
            ref.add(abs_path)

        if i % 200 == 0:
            host_log.info(f"Processed torrent file lists: {i}/{len(hashes)}")

    host_log.info(f"Referenced file paths collected: {len(ref)}")
    return ref


def remote_find_candidates_with_sizes(
    host_log: HostAdapter,
    ssh: SSHConfig,
    roots: List[str],
    min_age_minutes: int,
    exclude_contains: List[str],
) -> Dict[str, int]:
    """
    Remote 'find' returning a dict {path: size_bytes}.
    Uses NUL delimiters to be safe with spaces/special characters.
    Output format per root: path\\0size\\0path\\0size\\0...
    """
    all_files: Dict[str, int] = {}

    for root in roots:
        root_q = shlex.quote(root)
        age_expr = f"-mmin +{min_age_minutes}" if min_age_minutes > 0 else ""

        script = f"""\
set -euo pipefail
ROOT={root_q}
if [ ! -d "$ROOT" ]; then
  echo "WARN: root does not exist: $ROOT" >&2
  exit 0
fi

# Output: path\\0size\\0path\\0size...
find "$ROOT" -type f {age_expr} -printf '%p\\0%s\\0'
"""

        rc, out, err = ssh_run(ssh, stdin_bytes=script.encode("utf-8"), timeout=1800)
        if rc != 0:
            host_log.error(
                f"Remote find failed for root={root} rc={rc} err={err.decode('utf-8', 'replace')[:4000]}"
            )
            continue

        parts = out.split(b"\x00")
        files_for_root: Dict[str, int] = {}

        it = iter(parts)
        for p in it:
            if not p:
                continue
            s = next(it, b"0")
            path = p.decode("utf-8", errors="replace")
            try:
                size = int(s.decode("utf-8", errors="replace") or "0")
            except Exception:
                size = 0
            files_for_root[path] = size

        # exclusions
        if exclude_contains:
            before = len(files_for_root)
            files_for_root = {
                k: v for k, v in files_for_root.items() if not any(x in k for x in exclude_contains)
            }
            host_log.info(f"Root {root}: candidates {before} -> {len(files_for_root)} after excludes")
        else:
            host_log.info(f"Root {root}: candidates {len(files_for_root)}")

        all_files.update(files_for_root)

    return all_files


def compute_orphans(candidates: List[str], referenced: Set[str]) -> List[str]:
    # exact path match (absolute)
    return [f for f in candidates if normalize_path(f) not in referenced]


def remote_apply_actions(
    host_log: HostAdapter,
    ssh: SSHConfig,
    mode: str,
    roots: List[str],
    trash_subdir: str,
    trash_retention_days: int,
    clean_empty_dirs: bool,
    orphan_files: List[str],
) -> Tuple[int, int]:
    """
    Returns: (action_errors, affected_files_count)
    """
    if mode == "dry-run":
        for f in orphan_files[:200]:
            host_log.info(f"[DRY] orphan: {f}")
        if len(orphan_files) > 200:
            host_log.info(f"[DRY] ... plus {len(orphan_files) - 200} more")
        return 0, len(orphan_files)

    # Remote actions are implemented using a remote python3 snippet for robustness.
    roots_json = json.dumps(roots)
    files_json = json.dumps(orphan_files)

    remote = f"""\
set -euo pipefail

MODE={shlex.quote(mode)}
TRASH_SUBDIR={shlex.quote(trash_subdir)}
ROOTS_JSON={shlex.quote(roots_json)}
FILES_JSON={shlex.quote(files_json)}
RETENTION_DAYS={int(trash_retention_days)}
CLEAN_EMPTY_DIRS={1 if clean_empty_dirs else 0}

python3 - <<'PY'
import json, os, sys, subprocess, datetime

mode=os.environ["MODE"]
trash_subdir=os.environ["TRASH_SUBDIR"]
roots=json.loads(os.environ["ROOTS_JSON"])
files=json.loads(os.environ["FILES_JSON"])
retention=int(os.environ["RETENTION_DAYS"])
clean_empty=bool(int(os.environ["CLEAN_EMPTY_DIRS"]))

today=datetime.date.today().isoformat()

def find_root(path: str):
    matches=[r for r in roots if path.startswith(r.rstrip('/') + '/')]
    if not matches:
        return None
    return sorted(matches, key=len, reverse=True)[0]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

errors=0
moved=0
deleted=0

for f in files:
    if mode=="delete":
        try:
            os.remove(f)
            deleted += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            errors += 1
            print(f"ERR delete {f}: {e}", file=sys.stderr)

    elif mode=="trash":
        root=find_root(f)
        if not root:
            errors += 1
            print(f"ERR trash {f}: could not map to any root", file=sys.stderr)
            continue
        rel=f[len(root.rstrip('/') + '/'):]

        trash_base=os.path.join(root.rstrip('/'), trash_subdir, today)
        dest=os.path.join(trash_base, rel)
        ensure_dir(os.path.dirname(dest))

        try:
            os.rename(f, dest)  # same filesystem => fast
            moved += 1
        except OSError:
            # fallback to mv (handles cross-device)
            try:
                ensure_dir(os.path.dirname(dest))
                subprocess.check_call(["mv", "--", f, dest])
                moved += 1
            except Exception as e:
                errors += 1
                print(f"ERR trash {f} -> {dest}: {e}", file=sys.stderr)
    else:
        raise SystemExit(f"unknown mode {mode}")

# Empty dir cleanup
if clean_empty:
    for r in roots:
        if os.path.isdir(r):
            subprocess.call(["find", r, "-type", "d", "-empty", "-delete"])

# Trash retention cleanup
if mode=="trash" and retention>0:
    for r in roots:
        trash_root=os.path.join(r.rstrip('/'), trash_subdir)
        if os.path.isdir(trash_root):
            subprocess.call(["find", trash_root, "-type", "f", "-mtime", f"+{retention}", "-delete"])
            subprocess.call(["find", trash_root, "-type", "d", "-empty", "-delete"])

print(f"RESULT moved={moved} deleted={deleted} errors={errors}")
PY
"""

    rc, out, err = ssh_run(ssh, stdin_bytes=remote.encode("utf-8"), timeout=3600)
    if rc != 0:
        host_log.error(f"Remote action failed rc={rc} err={err.decode('utf-8','replace')[:4000]}")
        return 1, len(orphan_files)

    out_s = out.decode("utf-8", errors="replace").strip()
    err_s = err.decode("utf-8", errors="replace").strip()
    if err_s:
        host_log.warning(f"Remote stderr (truncated): {err_s[:4000]}")

    host_log.info(f"Remote action output: {out_s}")

    errors = 0
    for token in out_s.split():
        if token.startswith("errors="):
            try:
                errors = int(token.split("=", 1)[1])
            except Exception:
                pass

    return errors, len(orphan_files)


# -----------------------------
# Config parsing
# -----------------------------
def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    hosts: List[HostConfig] = []

    for h in raw.get("hosts", []):
        qb = QBConfig(
            url=h["qb"]["url"],
            username=h["qb"]["username"],
            password=h["qb"]["password"],
            verify_tls=bool(h["qb"].get("verify_tls", True)),
            timeout_seconds=int(h["qb"].get("timeout_seconds", 15)),
        )
        ssh = SSHConfig(
            host=h["ssh"]["host"],
            user=h["ssh"]["user"],
            port=int(h["ssh"].get("port", 22)),
            key_path=h["ssh"].get("key_path"),
            strict_host_key_checking=h["ssh"].get("strict_host_key_checking", "accept-new"),
            known_hosts_file=h["ssh"].get("known_hosts_file"),
        )

        hosts.append(
            HostConfig(
                name=h.get("name") or h["ssh"]["host"],
                qb=qb,
                ssh=ssh,
                download_roots=h.get("download_roots"),
                auto_roots=bool(h.get("auto_roots", True)),
                exclude_paths_containing=h.get("exclude_paths_containing"),
            )
        )

    cfg = AppConfig(
        hosts=hosts,
        min_age_minutes=int(raw.get("min_age_minutes", 240)),
        mode=str(raw.get("mode", "dry-run")),
        clean_empty_dirs=bool(raw.get("clean_empty_dirs", True)),
        trash_subdir=str(raw.get("trash_subdir", ".trash/qb_orphans")),
        trash_retention_days=int(raw.get("trash_retention_days", 14)),
        out_dir=str(raw.get("out_dir", "./runs")),
        exclude_paths_containing=raw.get("exclude_paths_containing", ["/.trash/"]),
    )
    return cfg


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--mode", choices=["dry-run", "trash", "delete"], help="Override mode")
    ap.add_argument("--min-age-minutes", type=int, help="Override min age filter")
    ap.add_argument("--out-dir", help="Override output directory")
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.mode:
        cfg.mode = args.mode
    if args.min_age_minutes is not None:
        cfg.min_age_minutes = args.min_age_minutes
    if args.out_dir:
        cfg.out_dir = args.out_dir

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / run_id
    logger = setup_logging(out_dir, args.log_level)
    base_log = HostAdapter(logger, {"host": "-"})

    base_log.info(
        f"Starting run mode={cfg.mode} min_age_minutes={cfg.min_age_minutes} hosts={len(cfg.hosts)}"
    )
    base_log.info(f"Output directory: {out_dir}")

    summary_path = out_dir / "summary.jsonl"

    overall_errors = 0

    # NEW: global summary accumulators
    total_orphans_count = 0
    total_orphans_bytes = 0
    total_candidates_count = 0
    total_candidates_bytes = 0

    per_host_summary: List[Dict[str, object]] = []

    for host in cfg.hosts:
        host_log = HostAdapter(logger, {"host": host.name})
        host_log.info("---- host start ----")

        # Per-host log file (extra)
        host_log_file = out_dir / f"{host.name}.log"
        fh = logging.FileHandler(host_log_file, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s host=%(host)s %(message)s",
                "%Y-%m-%dT%H:%M:%S%z",
            )
        )
        logger.addHandler(fh)

        t0 = time.time()

        # Per-host accumulators
        host_candidates_count = 0
        host_candidates_bytes = 0
        host_orphans_count = 0
        host_orphans_bytes = 0
        host_action_errors = 0

        try:
            qb = QBClient(host.qb)
            host_log.info(f"Logging into qB: {host.qb.url}")
            qb.login()

            torrents = qb.torrents_info()
            host_log.info(f"torrents/info count={len(torrents)}")

            roots = derive_scan_roots(host, torrents)
            host_log.info(f"scan_roots={roots}")

            referenced = build_referenced_paths(host_log, qb, torrents)

            # Merge global + per-host excludes
            global_ex = cfg.exclude_paths_containing or []
            host_ex = host.exclude_paths_containing or []
            exclude_contains = global_ex + host_ex
            if exclude_contains:
                host_log.info(f"exclude_contains={exclude_contains}")

            candidates_map = remote_find_candidates_with_sizes(
                host_log=host_log,
                ssh=host.ssh,
                roots=roots,
                min_age_minutes=cfg.min_age_minutes,
                exclude_contains=exclude_contains,
            )

            host_candidates_count = len(candidates_map)
            host_candidates_bytes = sum(candidates_map.values())
            host_log.info(f"remote candidates total={host_candidates_count}")
            host_log.info(f"candidates bytes={host_candidates_bytes}")

            orphans = compute_orphans(list(candidates_map.keys()), referenced)
            host_orphans_count = len(orphans)
            host_orphans_bytes = sum(candidates_map.get(p, 0) for p in orphans)

            host_log.info(f"orphans found={host_orphans_count}")
            host_log.info(f"orphans bytes={host_orphans_bytes} ({bytes_to_gib(host_orphans_bytes):.2f} GB)")

            action_errors, affected = remote_apply_actions(
                host_log=host_log,
                ssh=host.ssh,
                mode=cfg.mode,
                roots=roots,
                trash_subdir=cfg.trash_subdir,
                trash_retention_days=cfg.trash_retention_days,
                clean_empty_dirs=cfg.clean_empty_dirs,
                orphan_files=orphans,
            )
            host_action_errors = action_errors

            dur = round(time.time() - t0, 2)
            host_log.info(
                f"Done host duration_s={dur} affected={affected} action_errors={action_errors}"
            )

            # NEW: per-host summary block
            host_log.info(
                f"SUMMARY host={host.name} mode={cfg.mode} "
                f"orphans={host_orphans_count} size={bytes_to_gib(host_orphans_bytes):.2f} GB "
                f"candidates={host_candidates_count} candidates_size={bytes_to_gib(host_candidates_bytes):.2f} GB"
            )

            record = {
                "host": host.name,
                "mode": cfg.mode,
                "scan_roots": roots,
                "min_age_minutes": cfg.min_age_minutes,
                "candidates": host_candidates_count,
                "candidates_bytes": host_candidates_bytes,
                "orphans": host_orphans_count,
                "orphans_bytes": host_orphans_bytes,
                "action_errors": action_errors,
                "duration_s": dur,
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            if not summary_path.exists():
                summary_path.write_text("", encoding="utf-8")
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if action_errors:
                overall_errors += 1

        except Exception as e:
            overall_errors += 1
            host_log.exception(f"Host failed: {e}")
            record = {
                "host": host.name,
                "mode": cfg.mode,
                "error": str(e),
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            if not summary_path.exists():
                summary_path.write_text("", encoding="utf-8")
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        finally:
            # accumulate totals even if host partially worked
            total_candidates_count += host_candidates_count
            total_candidates_bytes += host_candidates_bytes
            total_orphans_count += host_orphans_count
            total_orphans_bytes += host_orphans_bytes

            per_host_summary.append(
                {
                    "host": host.name,
                    "mode": cfg.mode,
                    "orphans_count": host_orphans_count,
                    "orphans_gib": round(bytes_to_gib(host_orphans_bytes), 2),
                    "candidates_count": host_candidates_count,
                    "candidates_gib": round(bytes_to_gib(host_candidates_bytes), 2),
                    "action_errors": host_action_errors,
                }
            )

            logger.removeHandler(fh)
            fh.close()
            host_log.info("---- host end ----")
            base_log.info(f"Host finished: {host.name}")

    # NEW: final summary block (overall)
    base_log.info("========================================")
    base_log.info("FINAL SUMMARY")
    base_log.info(f"Mode: {cfg.mode}")
    base_log.info(f"Hosts: {len(cfg.hosts)} | Errors: {overall_errors}")
    base_log.info(
        f"Total candidates: {total_candidates_count} | {bytes_to_gib(total_candidates_bytes):.2f} GB"
    )
    base_log.info(
        f"Total orphans:    {total_orphans_count} | {bytes_to_gib(total_orphans_bytes):.2f} GB"
    )
    base_log.info("Per-host:")
    for row in per_host_summary:
        base_log.info(
            f"- {row['host']}: orphans={row['orphans_count']} "
            f"({row['orphans_gib']:.2f} GB), "
            f"candidates={row['candidates_count']} "
            f"({row['candidates_gib']:.2f} GB), "
            f"errors={row['action_errors']}"
        )
    base_log.info("========================================")

    base_log.info(f"Run complete. errors={overall_errors}")
    base_log.info(f"Summary: {summary_path}")
    return 1 if overall_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
