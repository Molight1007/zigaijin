#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cloud.py — MathPilot 云端测试通道（本地驱动，纯标准库）

工作流
    ① 本地改代码（主仓 = 唯一真源，服务器只做部署目标，禁止反向漂移）
    ② python tools/cloud.py push            增量推代码（约 30 MB，秒级）
    ③ python tools/cloud.py run <题单> [选项] 服务器 tmux 启动评测
    ④ python tools/cloud.py stat            看进度 / 内存 / lean 进程数
    ⑤ python tools/cloud.py pull            拉回结果（jsonl + 日志 + 报告）
    ⑥ 本地分析（读 diag / trace / verdicts / mathlib_usage_stats）→ 改代码 → 回 ②

一次性准备
    python tools/cloud.py doctor            自检：密钥、连通性、远端环境
    python tools/cloud.py push-assets       只做一次：传 Lean zip + 闭包（约 1.54 GB）

设计约束（本机实测）
    - 本机无 sshpass / rsync ⇒ 必须 SSH 密钥登录，同步走 tarfile + ssh/scp
    - 服务器是 Ubuntu 24.04；push 用显式白名单，绝不带大件与 .env

依赖：仅 Python 标准库（tarfile / subprocess / pathlib）
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import posixpath
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

# ---------------------------------------------------------------- 同步白名单
# 只推这些（代码 + 小脚本）。目录会递归；含 * 的按 glob 展开。
PUSH_ITEMS = [
    "agent",
    # ★★ 2026-10-09 补：`contracts/` 与 `cli/` 是 2026-10-07/08 架构解耦新建的两个包，
    #    此前漏在白名单外 ⇒ **云端根本没有 contracts/** ⇒
    #      · `agent/switch_registry.py`（shim）import 失败 → `agent/base.py:9` 的兜底
    #        `from switch_registry import …` 也失败 ⇒ `run_eval` 三级兜底全崩；
    #      · `agent/base.py:22` 的 `from contracts.types import Budget, Finding, BugReport`
    #        是**无兜底的硬依赖**，缺 contracts 时整链直接 ImportError。
    #    ⚠ 更隐蔽的是：`tools/cloud.py verify` 与 push **共用这份白名单**，
    #      所以漏包时 verify 仍报“服务器版本 == 本地版本（逐字节一致）” —— **假绿**。
    #    教训：往仓库新增**顶层包**时，必须同步这里，否则云端的“一致”是假的。
    "contracts",
    # ★★ 2026-10-09 补：原 `tools/` 下的 12 个「调用 agent 的 CLI 入口」已迁入 `cli/`。
    #    不补它 → 云端残留旧的 `tools/*.py`，且 `cli/*` 缺失（本地/远端分叉）。
    "cli",
    "prompts",
    "utils",
    "tools",
    "tests",
    # ★★ 2026-10-09 修：原为精确文件名 `run_eval.py` / `run_research.py` / `user_agent.py`，
    #    导致**根目录的 `__` 拆分模块全部漏传**（`run_eval__scoring.py` / `run_eval__engine.py`
    #    / `run_eval__classify.py` / `user_agent__config.py` / `user_agent__config_base.py`），
    #    云端报 `ModuleNotFoundError: No module named 'user_agent__config'`。
    #    改成 glob，覆盖主文件 + 其全部拆分模块（不会误匹配 `.pytmp_*`，它们以点开头）。
    "run_eval*.py",
    "run_research*.py",
    "user_agent*.py",
    "requirements.txt",
    "lean-toolchain",
    "SETUP.md",
    "ripgrep-14.1.1-x86_64-unknown-linux-musl",
    "题库",
    "formal-imo",
    "imo_steps",
    "sample_data",
    # ★ deploy/ 下的**脚本与配置**必须参与同步 —— 它们决定测试口径
    #   （run_112.py 的 CLI 参数、cloud_run.env 的环境映射）。
    #   但 deploy/ 同时存着 831MB zip 与 679MB 闭包，故用 glob 精确挑选；
    #   大件一律走 push-assets。
    "deploy/*.sh",
    "deploy/*.py",
    "deploy/*.env",
]

# 永不推送（体积大 / 含密钥 / 本地产物 / 生成物）
NEVER_PUSH = {
    ".env", ".git", ".venv", "__pycache__", ".pytest_cache", ".workbuddy", ".codebuddy",
    "data", "vendor", "lean下载版", "node_modules", "results", "测试结果", "output",
    "docs", "论文实践", "与lean相关的插件", "题库备份", "papers", "superhuman",
    # ★ 2026-10-09：`scripts/` 是开发用一次性分析脚本（analyze_timeout_0915 / attribution_0915），
    #    全仓**无任何运行时引用**，同名的旧副本还留在 tools/ 里 ⇒ 不必参与代码同步。
    #    显式排除它，好让 `_whitelist_coverage_gaps()` 保持"零告警"的判别力。
    "scripts",
    "ripgrep-14.1.1-x86_64-pc-windows-msvc",
    # 大资产 / 生成物
    "mathlib-olean", "lean-cache", "closure-full",
}
# 按相对路径精确排除
NEVER_PUSH_REL = {
    # ★ setup_lean.sh 在**服务器上**生成 lean-env.sh，内含服务器绝对路径。
    #   推上去会被 Windows 路径覆盖，直接毁掉 LEAN_PATH —— 必须排除。
    "deploy/lean-env.sh",
    "deploy/.known_hosts",     # 本机 ssh 状态
}
NEVER_PUSH_SUFFIX = (".pyc", ".docx", ".zip", ".tar.gz")


def iter_push_files():
    """展开白名单 → 产出 (绝对路径, 相对仓库根的 posix 路径)。push 与 verify 共用。"""
    seen: set[str] = set()
    for item in PUSH_ITEMS:
        matches = sorted(ROOT.glob(item)) if any(c in item for c in "*?[") \
            else [ROOT / item]
        for p in matches:
            if not p.exists():
                continue
            for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
                if not f.is_file():
                    continue
                try:
                    rel = f.relative_to(ROOT).as_posix()
                except ValueError:
                    continue
                if rel in seen or rel in NEVER_PUSH_REL:
                    continue
                if any(part in NEVER_PUSH for part in f.parts):
                    continue
                # deploy 下的 zip / 闭包不参与代码同步
                if rel.startswith("deploy/") and (
                        f.suffix == ".zip" or "mathlib-olean" in f.parts):
                    continue
                if f.suffix in NEVER_PUSH_SUFFIX:
                    continue
                seen.add(rel)
                yield f, rel


# 大资产：只走 push-assets
ASSET_ITEMS = [
    ("deploy/lean-4.31.0-linux.zip", "file"),
    ("deploy/mathlib-olean", "dir"),
]

# 仓库外的资产：官方非形式化语料（LeanSearch 离线语义检索用）
# ⚠️ 缺了它 Lsv2Corpus 会**静默**退化成"扫源码 + 关键词打分"的弱后端
CORPUS_LOCAL = Path(r"D:\leansearch_upstream\mathlib_informal_v4.16.0\data.jsonl")
CORPUS_REMOTE = "data/lsv2/mathlib_informal_v4.16.0.jsonl"

# 需要从服务器拉回的路径（相对远端仓库根）

# ---------------------------------------------------------------- 基础工具
def die(msg: str, code: int = 1):
    print(f"\n[错误] {msg}", file=sys.stderr)
    raise SystemExit(code)


def info(msg: str):
    print(f"  {msg}")


def head(msg: str):
    print(f"\n== {msg} ==")


def load_env() -> dict:
    """从 .env 读服务器配置（不打印任何密钥）。"""
    cfg = {}
    if not ENV_FILE.is_file():
        die(f"找不到 {ENV_FILE}")
    for raw in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip()
    # SSH 别名（推荐）：配了就从 ~/.ssh/config 取一切，无需 HOST/USER/KEY
    if not cfg.get("MATH_SERVER_SSH_ALIAS"):
        for k in ("MATH_SERVER_HOST", "MATH_SERVER_USER"):
            if not cfg.get(k):
                die(f".env 缺少 {k}（或改用 MATH_SERVER_SSH_ALIAS=mathpilot）")
    cfg.setdefault("MATH_SERVER_PORT", "22")
    if not cfg.get("MATH_SERVER_REMOTE_DIR"):
        user = cfg.get("MATH_SERVER_USER") or os.environ.get("USER") or "ubuntu"
        cfg["MATH_SERVER_REMOTE_DIR"] = f"/home/{user}/mathpilot"
    return cfg


def ssh_target(cfg: dict) -> str:
    """远端目标串。配了 MATH_SERVER_SSH_ALIAS 就用别名（推荐，见 ~/.ssh/config）。"""
    return cfg.get("MATH_SERVER_SSH_ALIAS") or \
        f"{cfg['MATH_SERVER_USER']}@{cfg['MATH_SERVER_HOST']}"


# known_hosts 落到仓库内，避免 ssh 去读/写 ~/.ssh
KNOWN_HOSTS = ROOT / "deploy" / ".known_hosts"


def _isolate_ssh(cfg: dict) -> list[str]:
    """让 ssh / scp 完全不触碰 ~/.ssh。

    背景（2026-09-20 实测）：后台任务会被沙箱拦截对 C:\\Users\\<user>\\.ssh 的读取，
    ssh 因此报「密钥登录未配置」。注意 ssh 默认会读 config、contact agent、
    读写 known_hosts —— 三处都要改掉，只加 -i 是不够的。
    """
    KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    kh = str(KNOWN_HOSTS).replace("\\", "/")
    o = ["-o", "IdentityAgent=none",
         "-o", f"UserKnownHostsFile={kh}",
         "-o", "StrictHostKeyChecking=accept-new"]
    if not cfg.get("MATH_SERVER_SSH_ALIAS"):
        o = ["-F", "none"] + o          # 不用别名时，连 config 也不读
    return o


def ssh_opts(cfg: dict) -> list[str]:
    """ssh 选项。"""
    o = _isolate_ssh(cfg)
    if not cfg.get("MATH_SERVER_SSH_ALIAS"):
        o += ["-p", str(cfg["MATH_SERVER_PORT"])]
    o += ["-o", "BatchMode=yes",            # 密钥没配好就立刻失败，不挂起等密码
          "-o", "ConnectTimeout=15",
          "-o", "ServerAliveInterval=30"]
    if cfg.get("MATH_SERVER_KEY"):
        o += ["-i", cfg["MATH_SERVER_KEY"]]
    return o


def scp_opts(cfg: dict) -> list[str]:
    """scp 选项（端口是 -P 大写，与 ssh 的 -p 不同）。"""
    o = _isolate_ssh(cfg)
    if not cfg.get("MATH_SERVER_SSH_ALIAS"):
        o += ["-P", str(cfg["MATH_SERVER_PORT"])]
    o += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if cfg.get("MATH_SERVER_KEY"):
        o += ["-i", cfg["MATH_SERVER_KEY"]]
    return o


def ssh_base(cfg: dict) -> list[str]:
    return ["ssh"] + ssh_opts(cfg) + [ssh_target(cfg)]


def ssh_run(cfg: dict, cmd: str, *, check=True, quiet=False) -> subprocess.CompletedProcess:
    full = ssh_base(cfg) + [cmd]
    if not quiet:
        info("$ " + cmd[:160] + ("…" if len(cmd) > 160 else ""))
    r = subprocess.run(full, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        if "Permission denied" in err or "BatchMode" in err:
            die("SSH 密钥登录未配置。见 `python tools/cloud.py doctor` 的提示。")
        die(f"远端命令失败（rc={r.returncode}）：\n{err[:800]}")
    return r


def scp_put(cfg: dict, local: Path, remote: str, *, recursive=False) -> None:
    cmd = ["scp", "-p"] + scp_opts(cfg)
    if recursive:
        cmd += ["-r"]
    cmd += [str(local), f"{ssh_target(cfg)}:{remote}"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        die(f"scp 上传失败：{(r.stderr or '').strip()[:600]}")


def scp_get(cfg: dict, remote: str, local: Path, *, recursive=False) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-p"] + scp_opts(cfg)
    if recursive:
        cmd += ["-r"]
    cmd += [f"{ssh_target(cfg)}:{remote}", str(local)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


# ---- 可执行位 -----------------------------------------------------------------
# ⚠️ Windows 没有可执行位概念，`tar` 打包带不上 ⇒ 上传后二进制会 Permission denied。
#   实测踩坑：`ripgrep-*/rg` 与 `tools/lean_local/bin/rg` 上传后都是 -rw-rw-r--。
#   git 也帮不上：本机 core.fileMode=false，`git ls-files -s` 显示 100755 的文件为 0 个。
#   ⇒ 只能按模式恢复。
PERM_EXEC_GLOBS = [
    "deploy/*.sh",
    "*.sh",
    "ripgrep-*/rg",
    "tools/lean_local/bin/rg",
]


def fix_perms(cfg: dict) -> None:
    """push 解包后恢复可执行位（幂等）。"""
    remote = cfg["MATH_SERVER_REMOTE_DIR"]
    cmds = [f"chmod +x {g} 2>/dev/null || true" for g in PERM_EXEC_GLOBS]
    # 兜底：任何名为 rg 的文件
    cmds.append("find . -type f -name rg -exec chmod +x {} + 2>/dev/null || true")
    cmds.append("echo perms_ok")
    ssh_run(cfg, f"cd {remote} && " + " ; ".join(cmds), quiet=True)


# ---------------------------------------------------------------- doctor
def cmd_doctor(cfg: dict, args) -> int:
    head("1 本机")
    info(f"仓库根      {ROOT}")
    info(f"Python      {sys.version.split()[0]}")
    for t in ("ssh", "scp"):
        import shutil
        info(f"{t:<11} {shutil.which(t) or '缺失！'}")

    head("2 SSH 配置（不含密钥）")
    if cfg.get("MATH_SERVER_SSH_ALIAS"):
        info(f"使用别名   {cfg['MATH_SERVER_SSH_ALIAS']}（取自 ~/.ssh/config）")
    else:
        info(f"直连       {ssh_target(cfg)}:{cfg['MATH_SERVER_PORT']}")
        if cfg.get("MATH_SERVER_KEY"):
            info(f"密钥       {cfg['MATH_SERVER_KEY']}")
    info(f"远端目录   {cfg['MATH_SERVER_REMOTE_DIR']}")
    info(f"实际命令   ssh {' '.join(ssh_opts(cfg))} {ssh_target(cfg)} …")

    head("3 密钥登录测试")
    r = ssh_run(cfg, "echo OK", check=False, quiet=True)
    if r.returncode != 0:
        info("❌ 密钥登录不通。本机没有 sshpass，密码无法自动化，请按下面做一次：")
        print("""
     # ① 本机生成密钥（已有可跳过；不要设 passphrase）
     ssh-keygen -t ed25519 -f %USERPROFILE%\\.ssh\\id_ed25519 -N ""

     # ② 把公钥装到服务器（会提示输一次密码）
     type %USERPROFILE%\\.ssh\\id_ed25519.pub | ssh -p PORT USER@HOST ^
       "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

     # ③ 复验
     ssh -p PORT USER@HOST echo OK
""".rstrip())
        info("（PowerShell 把 %USERPROFILE% 换成 $env:USERPROFILE，^ 换成 `）")
        return 1
    info("✅ 密钥登录正常")

    head("4 远端环境")
    for label, cmd in [
        ("内核", "uname -srm"),
        ("发行版", "(. /etc/os-release && echo $PRETTY_NAME)"),
        ("内存", "free -h | awk '/Mem:/{print $2\" 总 / \"$7\" 可用\"}'"),
        ("swap", "(swapon --show=SIZE --noheadings | tr '\\n' ' ') || echo 无"),
        ("系统盘", "df -h . | awk 'NR==2{print $2\" 总 / \"$4\" 可用\"}'"),
        ("CPU", "nproc"),
        # ★ 2026-10-04 修（假警报，属"代码地雷"）：原检查是 `command -v lean`，
        #   只查 **非交互 SSH 会话的 PATH** —— 而服务器上的 Lean 是靠
        #   `deploy/lean-env.sh` 激活的（导出 PATH + LD_LIBRARY_PATH + LEAN_PATH），
        #   启动器 `run_112.py` 会 source 它，但 `doctor` 自己不会。
        #   ⇒ 明明装好且能编译，却报"未安装"，误导判断（本次实测踩到）。
        #   现改为：**source lean-env.sh 后再查**，并附 Mathlib 闭包条数。
        ("lean", "cd %s 2>/dev/null && { . deploy/lean-env.sh 2>/dev/null; "
                 "(command -v lean >/dev/null && lean --version) || echo '未安装/未激活'; } "
                 "|| echo '仓库未上传，无法判 Lean'"
                 % cfg["MATH_SERVER_REMOTE_DIR"]),
        ("mathlib", "find %s/deploy/mathlib-olean/Mathlib -name '*.olean' 2>/dev/null "
                    "| wc -l | awk '{print $1\" 个 .olean\"}'"
                    % cfg["MATH_SERVER_REMOTE_DIR"]),
        ("仓库", f"[ -d {cfg['MATH_SERVER_REMOTE_DIR']} ] && echo 已就位 || echo 未上传"),
    ]:
        rr = ssh_run(cfg, cmd, check=False, quiet=True)
        info(f"{label:<8} {(rr.stdout or rr.stderr or '').strip()[:110]}")
    return 0


# ---------------------------------------------------------------- push
def build_tar(out: Path) -> tuple[int, int]:
    n = total = 0
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        for f, rel in iter_push_files():
            try:
                tf.add(f, arcname=rel)
                n += 1
                total += f.stat().st_size
            except OSError:
                pass
    return n, total


def cmd_push(cfg: dict, args) -> int:
    head("打包（白名单）")
    tmp = ROOT / "deploy" / f"_push_{int(time.time())}.tar.gz"
    n, total = build_tar(tmp)
    info(f"{n} 个文件 / {human(total)} → {tmp.name}")
    if n == 0:
        die("没有文件可推送，检查白名单")
    if args.dry_run:
        info("--dry-run：仅打包，不上传。已删除临时包。")
        tmp.unlink(missing_ok=True)
        return 0

    head("确保远端目录存在")
    ssh_run(cfg, f"mkdir -p {cfg['MATH_SERVER_REMOTE_DIR']}")

    head("上传并解包")
    remote_tar = f"/tmp/{tmp.name}"
    scp_put(cfg, tmp, remote_tar)
    ssh_run(cfg, f"cd {cfg['MATH_SERVER_REMOTE_DIR']} && tar xzf {remote_tar} "
                 f"&& rm -f {remote_tar}")
    tmp.unlink(missing_ok=True)

    head("恢复可执行位（Windows 无 exec 位，tar 带不上）")
    fix_perms(cfg)
    r = ssh_run(cfg, f"cd {cfg['MATH_SERVER_REMOTE_DIR']} && "
                     f"ls -l ripgrep-*/rg tools/lean_local/bin/rg 2>/dev/null | head -4",
               check=False, quiet=True)
    for ln in (r.stdout or "").strip().splitlines():
        info(ln)
    info("✅ 代码已同步")
    info("提示：.env 不会自动上传（已在服务器上，含密钥）")
    return 0


def cmd_push_assets(cfg: dict, args) -> int:
    head("大资产一次性同步（约 1.54 GB，仅需一次）")
    remote = cfg["MATH_SERVER_REMOTE_DIR"]
    ssh_run(cfg, f"mkdir -p {remote}/deploy")

    for rel, kind in ASSET_ITEMS:
        p = ROOT / rel
        if not p.exists():
            info(f"⏭ 跳过（本地不存在）：{rel}")
            continue
        if kind == "dir":
            info(f"上传目录 {rel}（{human(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))}）")
            scp_put(cfg, p, f"{remote}/deploy/", recursive=True)
        else:
            info(f"上传文件 {rel}（{human(p.stat().st_size)}）")
            scp_put(cfg, p, f"{remote}/deploy/")

    for s in ("setup_lean.sh", "verify_lean_linux.sh", "bootstrap_ubuntu.sh", "lean-env.sh"):
        sp = ROOT / "deploy" / s
        if sp.is_file():
            scp_put(cfg, sp, f"{remote}/deploy/")

    head("远端校验")
    ssh_run(cfg, f"cd {remote} && ls -la deploy/ | head -20")
    ssh_run(cfg, f"cd {remote} && bash deploy/bootstrap_ubuntu.sh || true")
    info("✅ 资产同步 + 初始化已执行（详见上方输出；报告见 verify_lean_linux_report.txt）")
    return 0


def cmd_push_corpus(cfg: dict, args) -> int:
    """上传仓库外的官方非形式化语料（194 MB，离线语义检索用）。"""
    remote = cfg["MATH_SERVER_REMOTE_DIR"]
    head("官方非形式化语料（LeanSearch 离线检索）")
    if not CORPUS_LOCAL.is_file():
        die(f"本地语料不存在：{CORPUS_LOCAL}\n"
            f"（该文件在仓库外，属独立资产；没有它检索会静默退化）")
    size = CORPUS_LOCAL.stat().st_size
    info(f"本地   {CORPUS_LOCAL}")
    info(f"       {human(size)}")
    info(f"远端   {remote}/{CORPUS_REMOTE}")

    dst_dir = posixpath.dirname(CORPUS_REMOTE)
    ssh_run(cfg, f"mkdir -p {remote}/{dst_dir}")
    scp_put(cfg, CORPUS_LOCAL, f"{remote}/{CORPUS_REMOTE}")
    r = ssh_run(cfg, f"cd {remote} && ls -l {CORPUS_REMOTE} && "
                     f"echo -n '行数 ' && wc -l < {CORPUS_REMOTE}", quiet=True)
    print((r.stdout or "").rstrip())
    info("✅ 语料就位（cloud_run.env 已指向该路径）")
    return 0


# ---------------------------------------------------------------- verify
def _local_manifest() -> tuple[dict, str]:
    """本地白名单文件的 {相对路径: sha256} + 汇总指纹。"""
    import hashlib
    m = {}
    for f, rel in iter_push_files():
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        m[rel] = h.hexdigest()
    lines = "\n".join(f"{m[k]}  {k}" for k in sorted(m))
    fp = hashlib.sha256(lines.encode("utf-8")).hexdigest()[:16]
    return m, fp


def _remote_filelist_cmd(remote: str) -> str:
    """生成"列出远端被跟踪文件"的命令（与白名单同源，含 glob 展开）。"""
    import shlex
    parts = []
    for item in PUSH_ITEMS:
        if any(c in item for c in "*?["):
            parent, _, pat = item.rpartition("/")
            parts.append(f"find {shlex.quote(parent)} -maxdepth 1 -type f "
                         f"-name {shlex.quote(pat)} 2>/dev/null")
        else:
            p = ROOT / item
            if p.is_dir():
                parts.append(f"find {shlex.quote(item)} -type f 2>/dev/null")
            else:
                parts.append(f"([ -f {shlex.quote(item)} ] && echo {shlex.quote(item)})")
    return f"cd {remote} && {{ " + "; ".join(parts) + "; } | sort"


def _whitelist_coverage_gaps() -> list[str]:
    """本地顶层包 / 根目录 .py 里，有哪些**没被 `PUSH_ITEMS` 覆盖**。

    ★ 为什么必须有这道检查（2026-10-09 实测教训）：`verify` 与 `push` **共用同一份
      白名单**，所以白名单漏了东西时，verify 仍会报「服务器版本 == 本地版本（逐字节一致）」
      —— **假绿**。同一类问题当天连踩两次：
        ① 架构解耦新建的 `contracts/` 与 `cli/` 两个包没进白名单 →
           云端 `import contracts` 失败、`run_eval` 三级兜底全崩；
        ② 白名单写的是精确文件名 `user_agent.py`/`run_eval.py` →
           根目录的 `__` 拆分模块（`user_agent__config.py` 等 5 个）全漏 →
           云端 `ModuleNotFoundError: No module named 'user_agent__config'`。
      ⇒ 新增顶层包或拆分模块时，这里会立刻把它们报出来。
    """
    globs = [i for i in PUSH_ITEMS if any(c in i for c in "*?[")]
    items = set(PUSH_ITEMS)

    def covered(rel: str) -> bool:
        return rel in items or any(fnmatch.fnmatch(rel, g) for g in globs)

    gaps: list[str] = []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if d.name.startswith(".") or d.name in NEVER_PUSH:
            continue
        if not any(d.rglob("*.py")):
            continue
        # 目录被覆盖，或该目录下的 .py 被 glob（如 deploy/*.py）覆盖，都算已覆盖
        if not (covered(d.name) or covered(d.name + "/x.py")):
            gaps.append("%s/（顶层包，不在白名单）" % d.name)
    for f in sorted(ROOT.glob("*.py")):
        if f.name.startswith("."):  # 跳过 .pytmp_* 之类临时文件
            continue
        if not covered(f.name):
            gaps.append(f.name)
    return gaps


def cmd_verify(cfg: dict, args) -> int:
    """校验「服务器上的代码 == 本地代码」。push 后必跑。"""
    remote = cfg["MATH_SERVER_REMOTE_DIR"]
    head("1 本地清单")
    local, fp = _local_manifest()
    info(f"{len(local)} 个文件（白名单），指纹 {fp}")

    manifest = ROOT / "deploy" / "_manifest.tsv"
    with open(manifest, "w", encoding="utf-8", newline="\n") as fh:
        for k in sorted(local):
            fh.write(f"{local[k]}  {k}\n")
    info(f"清单 {manifest.name}（sha256sum -c 格式）")

    head("2 上传清单并逐文件校验")
    scp_put(cfg, manifest, "/tmp/_mathpilot_manifest.tsv")
    r = ssh_run(cfg, f"cd {remote} && sha256sum -c /tmp/_mathpilot_manifest.tsv 2>&1",
                check=False, quiet=True)
    out = (r.stdout or "") + (r.stderr or "")
    ok = len(re.findall(r": OK$", out, re.M))
    bad = re.findall(r"^(.*?): FAILED$", out, re.M)
    missing = re.findall(r"^(.*?): (?:No such file|没有那个文件)", out, re.M)
    info(f"一致 {ok} / 不一致 {len(bad)} / 缺失 {len(missing)}")

    head("3 远端多余文件（本地已删但服务器还在的）")
    r = ssh_run(cfg, _remote_filelist_cmd(remote), check=False, quiet=True)
    rem = {ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()}
    # 服务器上由脚本生成的、本地产出不了的，不算多余
    rem -= {"deploy/lean-env.sh"}
    # Python 字节码缓存：服务器上跑过就会生成，不是"漂移"
    rem = {r for r in rem if "__pycache__" not in r and not r.endswith(".pyc")}
    extra = sorted(rem - set(local))
    if extra:
        info(f"⚠️ {len(extra)} 个：")
        for e in extra[:30]:
            info("   " + e)
        if len(extra) > 30:
            info(f"   …还有 {len(extra)-30} 个")
    else:
        info("✅ 无多余文件")

    head("白名单覆盖自检（防「verify 假绿」）")
    gaps = _whitelist_coverage_gaps()
    if gaps:
        info(f"⚠️ {len(gaps)} 处未被 PUSH_ITEMS 覆盖 —— 这些内容**根本不会被推送**，"
             f"云端缺它们时上面第 2 节仍会报「一致」（假绿）：")
        for g in gaps[:20]:
            info("   " + g)
    else:
        info("✅ 所有顶层包与根 .py 均已被白名单覆盖")

    head("结论")
    if not bad and not missing:
        info(f"✅ 服务器版本 == 本地版本（{len(local)} 文件逐字节一致）")
        info(f"   版本指纹 {fp}")
        if extra:
            info(f"   ⚠️ 但有 {len(extra)} 个远端残留（不影响哈希，建议 push 后复查）")
        return 0
    info("❌ 不一致 —— 先执行：python tools/cloud.py push")
    for b in bad[:20]:
        info("   内容不同 " + b)
    for m in missing[:20]:
        info("   服务器缺 " + m)
    return 1


# ---------------------------------------------------------------- run
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def cmd_run(cfg: dict, args) -> int:
    remote = cfg["MATH_SERVER_REMOTE_DIR"]
    name = args.name or f"run_{time.strftime('%m%d_%H%M')}"
    if not RUN_NAME_RE.match(name):
        die("run 名称只能用字母数字下划线连字符")
    launcher = args.launcher

    head("串行检查（禁止并发跑测）")
    r = ssh_run(cfg, "tmux ls 2>/dev/null || true", quiet=True)
    active = [l.split(":")[0] for l in (r.stdout or "").splitlines() if ":" in l]
    if active:
        die(f"服务器上已有 tmux 会话在跑：{', '.join(active)}\n"
            f"串行铁律：先等它结束，或手工 `tmux attach -t <名字>` 确认后再启新测试。")
    info("✅ 无其它会话")

    # 前置检查：启动器、venv、题单都在
    pre = (f"cd {remote} && "
           f"for f in {launcher} .venv/bin/python deploy/monitor.sh; do "
           f"[ -e \"$f\" ] && echo \"OK  $f\" || echo \"缺失 $f\"; done")
    r = ssh_run(cfg, pre, quiet=True)
    for ln in (r.stdout or "").strip().splitlines():
        info("  " + ln)
    if "缺失" in (r.stdout or ""):
        die("前置文件不全 → 先 python tools/cloud.py push（并确认 push-assets 已跑完）")

    eval_cmd = (f"cd {remote} && .venv/bin/python -u {launcher} --tag {name} "
                f"{args.launcher_args or ''}")
    mon_cmd = f"cd {remote} && bash deploy/monitor.sh logs/monitor_{name}.csv 5 0"

    print("\n即将在服务器上启动：")
    info(f"会话名     {name}（tmux 两个窗口：eval / mon）")
    info(f"评测       {eval_cmd}")
    info(f"资源采样   {mon_cmd}")
    info("           输出 logs/monitor_*.csv（内存/swap/lean 进程数/CPU 曲线）")
    if not args.yes:
        ans = input("\n确认启动？(yes/no) ").strip().lower()
        if ans not in ("y", "yes"):
            info("已取消。")
            return 0

    head("启动")
    cmd = (f"cd {remote} && tmux new-session -d -s {name} -n eval '{eval_cmd}' "
           f"&& sleep 2 && tmux new-window -t {name} -n mon '{mon_cmd}' "
           f"&& sleep 3 && tmux ls && ls -l logs/ 2>/dev/null | tail -5")
    r = ssh_run(cfg, cmd)
    print((r.stdout or "").rstrip())
    info("✅ 已启动。进度：python tools/cloud.py stat")
    info(f"    看日志：ssh … \"tmux attach -t {name}\"（Ctrl-b w 切窗口，Ctrl-b d 脱离）")
    return 0


def cmd_stat(cfg: dict, args) -> int:
    remote = cfg["MATH_SERVER_REMOTE_DIR"]
    head("会话")
    r = ssh_run(cfg, "tmux ls 2>/dev/null || echo '(无会话)'", quiet=True)
    print((r.stdout or "").strip())

    head("进度")
    cmd = (f"cd {remote} 2>/dev/null || exit 0; "
           "for f in results/cloud_*.jsonl; do "
           "[ -f \"$f\" ] || continue; "
           "n=$(wc -l < \"$f\"); echo \"$f  已完成 $n 题\"; done; "
           "ls -t logs/cloud_*.log 2>/dev/null | head -1 | "
           "while read l; do echo \"最新日志 $l\"; tail -n 3 \"$l\" | sed 's/^/    /'; done")
    r = ssh_run(cfg, cmd, quiet=True)
    print((r.stdout or "").rstrip())

    head("资源（实时）")
    cmd2 = ("free -h | sed -n '1,3p'; echo; "
            "echo \"lean 进程数: $(pgrep -c -x lean 2>/dev/null || echo 0)\"; echo; "
            f"cd {remote} && df -h . | tail -1")
    r = ssh_run(cfg, cmd2, quiet=True)
    print((r.stdout or "").rstrip())

    head("资源采样曲线（monitor CSV）")
    cmd3 = (f"cd {remote} && f=$(ls -t logs/monitor_*.csv 2>/dev/null | head -1); "
            "[ -n \"$f\" ] || { echo '（尚无 monitor CSV）'; exit 0; }; "
            "echo \"文件 $f   采样行数 $(($(wc -l < \"$f\") - 1))\"; "
            "echo \"表头: $(head -1 \"$f\")\"; "
            "echo \"最新: $(tail -1 \"$f\")\"; "
            "awk -F, 'NR>1{if(NR==2||$4+0<ma)ma=$4; if($8+0>ml)ml=$8; "
            "if($9+0>mr)mr=$9; if($5+0>mp)mp=$5} "
            "END{printf \"区间统计: 最低可用内存 %d MB | 最多 lean 进程 %d | "
            "lean 峰值RSS合计 %d MB | 内存占用峰值 %d%%\\n\", ma, ml, mr, mp}' \"$f\"")
    r = ssh_run(cfg, cmd3, quiet=True)
    print((r.stdout or "").rstrip())
    return 0


# ---------------------------------------------------------------- pull
def cmd_pull(cfg: dict, args) -> int:
    remote = cfg["MATH_SERVER_REMOTE_DIR"]
    dest = ROOT / "results" / "_cloud"
    dest.mkdir(parents=True, exist_ok=True)

    head("拉取结果")
    got = 0
    # results/ 整体：jsonl（17 顶层字段 + diag 69 键 + trace 111 条）
    #                + 日志(.log) + 环境快照(.env)
    r = ssh_run(cfg, f"ls {remote}/results/*.jsonl {remote}/results/*.log "
                     f"{remote}/results/*.env 2>/dev/null | wc -l", quiet=True)
    n = int((r.stdout or "0").strip() or 0)
    if n:
        scp_get(cfg, f"{remote}/results/", dest / "results", recursive=True)
        got += n
        info(f"results/              {n} 个文件（jsonl + log + env 快照）")
    else:
        info("results/ 暂无产物")

    # logs/ 整体：monitor_*.csv（资源曲线，jsonl 里没有）
    r = ssh_run(cfg, f"ls {remote}/logs/* 2>/dev/null | wc -l", quiet=True)
    n = int((r.stdout or "0").strip() or 0)
    if n:
        scp_get(cfg, f"{remote}/logs/", dest / "logs", recursive=True)
        got += n
        info(f"logs/                 {n} 个文件（含 monitor CSV 资源曲线）")
    else:
        info("logs/ 暂无产物")

    # 跨平台报告
    if scp_get(cfg, f"{remote}/verify_lean_linux_report.txt", dest / "verify_lean_linux_report.txt"):
        got += 1
        info("verify_lean_linux_report.txt")

    head("摘要")
    if not got:
        info("未拉到任何文件。检查服务器上是否已产出（python tools/cloud.py stat）。")
        return 1
    info(f"共 {got} 个文件 → {dest}")
    info("数据完整性（实测确认，无需再补）：")
    info("  逐题答题情况  results/*.jsonl  17 个顶层字段 + diag 69 个键")
    # 2026-10-02 起阶段数随删除同步更新（0_paper_pacer 已删 ⇒ 16→15）；
    # 2026-10-14 末段改为 `6_format`（6.5_audit_gate 已于五道链定案中删除），
    #   `1.1_type_understanding` 并入 2.6 题意理解 ⇒ 15 段；
    # ★ 2026-10-14（任务列表 #2）`1.2_theorem_hint` 下移为 2.7 体首动作 ⇒ 14 段，
    #   首段变为 `2.6_pre_audit`。
    info("  分步耗时      diag.stage_timers  14 个阶段（2.6_pre_audit … 6_format）")
    info("  分步叙事      trace  每题约 111 条")
    info("  资源曲线      logs/monitor_*.csv（内存/swap/lean 进程数/CPU）")
    info("  8 个分析脚本均以 jsonl 为输入（stage_timers / gen_test_report / diagnosis / "
         "attribution / time_allocation / analyze_errors / record_eval_report）")
    info("")
    info("下一步：出 Word 报告")
    info('  python tools/gen_test_report.py --results results/_cloud/results/<结果>.jsonl '
         '--testset <题单>.jsonl --core results/_112_core_table.jsonl '
         '--out-md "测试结果/xin测试结果/云端_<名>_' + time.strftime('%Y%m%d') + '.md"')
    info("  再用 D:\\mathpilot_work\\tools\\_mkdocx.py 转 docx")
    info("⚠️ 生成器两个坑：md 里禁止围栏代码块；内容里的竖线要换成 ∣(U+2223)")
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="MathPilot 云端测试通道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="自检：密钥、连通性、远端环境").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("push", help="增量推送代码（约 30 MB）")
    p.add_argument("--dry-run", action="store_true", help="只打包不上传")
    p.set_defaults(fn=cmd_push)

    sub.add_parser("push-assets", help="一次性传 Lean zip + 闭包（约 1.54 GB）"
                  ).set_defaults(fn=cmd_push_assets)
    sub.add_parser("push-corpus", help="上传仓库外的官方语料（194 MB）"
                  ).set_defaults(fn=cmd_push_corpus)
    sub.add_parser("verify", help="校验「服务器代码 == 本地代码」（push 后必跑）"
                  ).set_defaults(fn=cmd_verify)

    p = sub.add_parser("run", help="在服务器上启动评测（默认 112 题，含资源采样）")
    p.add_argument("--launcher", default="deploy/run_112.py",
                   help="远端启动器（默认 deploy/run_112.py）")
    p.add_argument("--launcher-args", default="", help="追加给启动器的参数")
    p.add_argument("--name", help="会话名/产物名，默认 run_MMDD_HHMM")
    p.add_argument("--yes", action="store_true", help="跳过确认")
    p.set_defaults(fn=cmd_run)

    sub.add_parser("stat", help="查看进度/内存/lean 进程数").set_defaults(fn=cmd_stat)
    sub.add_parser("pull", help="拉回结果到 results/_cloud/").set_defaults(fn=cmd_pull)

    args = ap.parse_args()
    cfg = load_env()
    return args.fn(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
