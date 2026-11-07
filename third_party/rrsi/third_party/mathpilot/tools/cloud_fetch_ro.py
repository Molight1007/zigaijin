#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读取数：把云端某轮评测的结果文件快照到**仓库外**目录。

硬约束（铁律）
--------------
- **只读远端**：仅执行 `ls` / `scp <远端> -> 本地`，绝不 push / 不写远端。
- **不写 `results/_cloud/`**：评测运行期间该目录属禁区（口径见 MEMORY）。
  默认落地 `D:\\mathpilot_work\\cloud_snap\\_cloud_snap_<时间戳>\\`，可用 `--dest` 覆盖。
- 不做任何代码同步、不 verify、不 kill 会话。

用法
----
  python tools/cloud_fetch_ro.py --tag run_2026-09-22
  python tools/cloud_fetch_ro.py --tag run_2026-09-22 --dest D:\\mathpilot_work\\cloud_snap\\_cloud_snap_now
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

KEY = r"C:\Users\35174\Downloads\workbuddy.pem"
HOST = "ubuntu@211.159.188.206"
REMOTE = "/home/ubuntu/mathpilot"


def run(cmd: list, timeout: int = 180):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def ssh(cmd: str, timeout: int = 60):
    return run(["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=15", HOST, cmd], timeout=timeout)


def fetch(tag: str, dest: str, with_monitor: bool = True) -> int:
    os.makedirs(dest, exist_ok=True)
    got, miss = [], []

    # 1) results/<tag>*  （.jsonl / .log / .arm.json / .env）
    r = ssh(f"ls -1 {REMOTE}/results/{tag}* 2>/dev/null")
    remote_files = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
    for rf in remote_files:
        try:
            rr = run(["scp", "-i", KEY, "-o", "StrictHostKeyChecking=no",
                      f"{HOST}:{rf}", dest], timeout=300)
            (got if rr.returncode == 0 else miss).append(os.path.basename(rf))
        except Exception as e:  # noqa: BLE001
            miss.append(f"{os.path.basename(rf)} ({e})")

    # 2) logs/monitor_<tag>.csv
    if with_monitor:
        mf = f"{REMOTE}/logs/monitor_{tag}.csv"
        rr = ssh(f"ls -1 {mf} 2>/dev/null")
        if rr.stdout.strip():
            try:
                r2 = run(["scp", "-i", KEY, "-o", "StrictHostKeyChecking=no",
                          f"{HOST}:{mf}", dest], timeout=300)
                (got if r2.returncode == 0 else miss).append(f"monitor_{tag}.csv")
            except Exception as e:  # noqa: BLE001
                miss.append(f"monitor_{tag}.csv ({e})")

    # 3) 运行态（只读判定）
    st = ssh("tmux ls 2>&1; echo '---'; pgrep -af 'run_112.py' 2>&1")
    write_text(os.path.join(dest, "_RUNSTATE.txt"),
               f"fetched_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\ntag: {tag}\n\n{st.stdout}\n")

    print(f"[fetch] dest = {dest}")
    print(f"[fetch] got  = {len(got)} 文件")
    for f in sorted(got):
        p = os.path.join(dest, f)
        print(f"         {os.path.getsize(p):>10d}  {f}")
    if miss:
        print(f"[fetch] 未取到 = {miss}")
    print("[fetch] 运行态：")
    print((st.stdout or "").strip())
    return 0 if got else 1


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="远端结果前缀，如 run_2026-09-22")
    ap.add_argument("--dest", default="")
    ap.add_argument("--no-monitor", action="store_true")
    a = ap.parse_args()
    dest = a.dest or os.path.join(r"C:\Users\35174",
                                 "_cloud_snap_" + time.strftime("%m%d_%H%M"))
    return fetch(a.tag, dest, with_monitor=not a.no_monitor)


if __name__ == "__main__":
    sys.exit(main())
