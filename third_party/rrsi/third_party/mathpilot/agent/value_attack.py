"""数值攻击验证器（C-lite，2026-09-03 老师拍板）。

拦截"蓝图/求解声称数值极值，实际错误"的系统性问题（009 实况：蓝图声称
最大值 4∛(85/98)≈3.815，真值 2∛(196/13)≈4.94——LLM 心算错值后全链路
执行+验证自洽，Lean unknown 拦不住）。

原理：对连续极值/不等式类题（题面含 S=表达式 + 变量约束 + 求 max/min），
从题面提取 Python 可执行的目标函数，**随机+边界采样攻击**——若采样点
得到的目标值突破声称极值 → 声称值不成立（确定性证伪，非 LLM 自评）。

拦截点：蓝图 merge / 候选答案里出现"极值 = 数值"时。
"""
import logging

logger = logging.getLogger("MathPilot")

# 声称极值触发词（中英）
# 数值表达式（声称的极值）

# 采样规模：准确率 vs 速度平衡（30k ≈ 009 找到 4.913/4.94）
_N_SAMPLES = 20000
# 证伪阈值：采样值 > 声称值 × (1+eps) 才算反例（防数值噪声误伤）
_EPS = 1e-6





def attack_value_claim(problem: str, claimed: float, direction: str,
                       func_src: str | None, n_samples: int = _N_SAMPLES
                       ) -> dict:
    """数值攻击声称值。

    func_src: LLM 生成的 Python 函数源码，形如
        def f(x):   # x 是变量列表
            ...用 x[0],x[1],... 计算目标值...
            return value
    调用约定：f 接受 list[float]（长度=变量数），返回 float；内部自含约束
    （不可行点返回 None/raise → 调用方跳过）。
    """
    if not func_src:
        return {"ok": True, "reason": "目标函数不可用，跳过攻击"}
    ns: dict = {}
    try:
        exec(func_src, ns)
        fn = ns.get("f")
        if fn is None:
            return {"ok": True, "reason": "函数源码无 f，跳过"}
    except Exception as e:  # noqa: BLE001
        return {"ok": True, "reason": f"函数编译失败: {str(e)[:80]}"}
    best = None
    try:
        for _ in range(n_samples):
            pt = None
            # 只使用函数自带的 sample_point；未提供时该点直接跳过。
            # ⚠ 本函数**没有**"n 元单纯形"兜底实现 —— 2026-09-20 更正：
            #   原注释"否则 n 元单纯形"与代码事实不符（误导排查方向）。
            sampler = ns.get("sample_point")
            try:
                if sampler:
                    pt = sampler()
                else:
                    raise TypeError
            except Exception:  # noqa: BLE001
                continue
            try:
                val = fn(list(pt))
            except Exception:  # noqa: BLE001  不可行点
                continue
            if val is None:
                continue
            # 2026-09-20 修复：聚合方向必须与声称方向一致。
            # 原实现恒取**最大值**，却在下方用 `best < claimed` 判 min 类声称
            # ⇒ 等于用最大值去证明"存在更小的值"，最小值声称几乎不可能被证伪，
            # 且返回的 found 与文案（"发现目标值 < 声称最小值"）自相矛盾。
            if direction == "min":
                if best is None or val < best:
                    best = val
            else:
                if best is None or val > best:
                    best = val
    except Exception as e:  # noqa: BLE001
        return {"ok": True, "reason": f"采样异常: {str(e)[:80]}"}
    if best is None:
        return {"ok": True,
                "reason": "采样无可行点（未提供 sample_point 或点均不可行），跳过"}
    # 声称"最大值=M"：找 S > M → 证伪
    if direction == "max":
        if best > claimed + _EPS:
            return {
                "ok": False, "claimed": claimed, "found": best,
                "reason": (f"数值攻击证伪：采样发现目标值 {best:.6f} > "
                           f"声称最大值 {claimed:.6f}——声称的极值不成立"),
            }
        return {"ok": True, "found": best, "reason": "采样未突破声称最大值"}
    # 声称"最小值=m"：找 S < m → 证伪
    if best < claimed - _EPS:
        return {
            "ok": False, "claimed": claimed, "found": best,
            "reason": (f"数值攻击证伪：采样发现目标值 {best:.6f} < "
                       f"声称最小值 {claimed:.6f}——声称的极值不成立"),
        }
    return {"ok": True, "found": best, "reason": "采样未突破声称最小值"}
