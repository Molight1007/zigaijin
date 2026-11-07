# -*- coding: utf-8 -*-
"""子目标「逐步求解」主循环（2026-10-08 批1-B增：从 agent/sub_goal_solver.py 的
SubGoalSolverAgent.run 抽出）。

抽出段（run 内顺序）：
  · solve_subgoals         —— `for sg in subgoals:` 全循环（复用命中 / 上下文准备 /
                              求解与失败重试 / 收尾落盘）与 `_ctx_inject_chars` 累计
  · sg_light_check_resolve —— S1-lite 子目标级 0-LLM 校验（失败带反馈重解一次）
  · sg_adaptive_recover    —— 子目标级「失败二选一」处置（重做 / 交上游重规划）

**纯搬迁**：段内源码逐字搬入（仅去缩进），调用点由 run 改为一行委托；两块自包含
子段（0-LLM 校验 / 自适应恢复）在本模块内按语义拆成 ≤150 的 helper。
`continue`（复用命中）仍在 `solve_subgoals` 的循环体内，未跨函数边界。

`solve_subgoals` 引用 sub_goal_solver 模块级 `_artifact_append` ⇒ **函数内惰性
import**（单一来源不复制）。参数 `self` 即 `SubGoalSolverAgent` 实例。
`solve_subgoals` 返回累计前序注入字符数 `_ctx_inject_chars`（run 后续 record 使用）。
"""
from __future__ import annotations

import time

# ★ 2026-10-15（用户「2 与 3 之间建一个环」）：求解期动态倒推。
#   backward 模块模块级**只依赖 stdlib**（对 prompts / blueprint_planner 的引用
#   全在函数体内惰性 import）⇒ 此处模块级 import 无循环风险。
from .sub_goal_solver__backward import sg_dynamic_backward  # noqa: F401
from .subgoal_result import cached_contract, cached_text, write_step



def sg_light_check_resolve(self, ctx, sg, step_result, subgoal_plan_summary,
                           prev_results) -> str:
    """S1-lite 子目标级 0-LLM 校验；未过则带反馈重解一次。返回 step_result。"""
    # S1-lite（2026-09-06 老师建议）：子目标级 0-LLM 校验前移——
    # 便宜且确定的校验先跑（截断/lean 代码片编译），过了才认结果，
    # 不让脏结果流进 merge 与后续子目标（验证-精炼下沉到子目标级）。
    # 校验全部 0-LLM；失败 → 带反馈重解一次；仍失败用原结果。
    # 2026-09-17 移除时间闸（原 `not ctx.gen_time_up() and _stage_left() > 90`）：
    # 时间无限，0-LLM 校验无条件执行（成本仅本地计算，无 LLM 开销）。
    if not step_result.startswith("[子目标"):
        _hint = self._subgoal_light_check(ctx, sg, step_result)
        if _hint:
            self.record(
                ctx, "subgoal_step",
                f"子目标 #{sg['id']}「{sg['title']}」0-LLM 校验未过: "
                f"{_hint[:80]}，带反馈重解一次")
            retry3 = self._solve_subgoal(
                ctx, sg, subgoal_plan_summary, prev_results,
                extra_hint=_hint)
            if retry3 and not retry3.startswith("[子目标"):
                # 重解结果仍疑似占位（二次空转）→ 显式失败标记，
                # 让 merge 明确知道该子目标无产出，不再用原占位糊弄。
                if (retry3 != step_result
                        and self._looks_placeholder(retry3)):
                    step_result = (
                        f"（子目标 #{sg['id']} 重解后仍为空转占位，"
                        "未产出有效结论；合并时请忽略并基于其余子目标求解）")
                else:
                    step_result = retry3
            else:
                self.record(
                    ctx, "subgoal_step",
                    f"子目标 #{sg['id']} 校验重解失败/超时，用原结果兜底")
                if self._looks_placeholder(step_result):
                    step_result = (
                        f"（子目标 #{sg['id']} 求解失败未产出有效结论，"
                        "请基于其余子目标完成合并）")
    return step_result


def sg_adaptive_recover(self, ctx, sg, step_result, subgoal_plan_summary,
                        prev_results) -> str:
    """子目标级「失败二选一」处置（重做 / 交上游重规划）。返回 step_result。"""
    # ── 2026-09-12 新增（用户要求）：子目标级"失败二选一"处置 ──────────
    # 判据（主审定）：
    #   · **优先「重做该子目标」**：代价 1 次短调用，且**不破坏已完成成果**；
    #     多数子目标失败是局部性的（算错 / 格式 / 占位）。
    #   · 仅当重做达上限仍失败，**或**失败原因指向规划本身（依赖前提
    #     不成立 / 缺少前置条件），才升级为「重规划后续子目标」——
    #     该操作代价高（后续子目标全部重解），故只在最后才用。
    #   · 无论如何最终**保留占位兜底放行**，绝不阻断 merge；
    #     已完成子目标一律保留、不参与重算（见上方 _done 复用）。
    # 稳妥性：真正的"重规划"**不在此循环内新建实现**，而是把失败原因写入
    # ctx.revise_feedback、并由上游既有重规划能力（_review_and_maybe_replan
    # / dag_replan_gate）处理，避免破坏 depends_on 顺序与阶段预算。
    # 2026-09-17 移除时间闸（原 `not ctx.gen_time_up() and _stage_left() > 150`）：
    # 时间无限，失败恢复无条件尝试（重做上限仍由下方 `_tries < 2` 控制，
    # 不存在无限循环）。
    if getattr(self.config, "subgoal_adaptive_recover", True):
        _failed_now = (self._looks_placeholder(step_result)
                       or step_result.startswith("[子目标")
                       or "未产出有效结论" in step_result
                       or "仍为空转占位" in step_result)
        if _failed_now:
            _tries = int(getattr(ctx, "_sg_recover_tries", 0) or 0)
            _plan_err = any(_k in str(step_result) for _k in
                            ("依赖", "前提", "缺少", "不成立", "无法确定"))
            if _tries < 2 and not _plan_err:
                try:
                    setattr(ctx, "_sg_recover_tries", _tries + 1)
                except Exception:  # noqa: BLE001
                    pass
                self.record(
                    ctx, "subgoal_recover",
                    f"子目标 #{sg['id']}「{sg['title']}」未产出有效结论 "
                    f"→ 重做该子目标（第 {_tries + 1}/2 次）")
                _retry2 = self._solve_subgoal(
                    ctx, sg, subgoal_plan_summary, prev_results,
                    extra_hint=(
                        "上一次该子目标未产出可用的结论（"
                        + str(step_result)[:120] +
                        "）。请**重新推导**本子目标，给出可独立核验的结论；"
                        "涉及数值运算请逐步写清算式与代入过程，便于复核。"))
                if (_retry2 and not _retry2.startswith("[子目标")
                        and not self._looks_placeholder(_retry2)):
                    step_result = _retry2
            else:
                # 升级：重规划**后续**子目标（只记录 + 交上游，不在循环内新建）
                self.record(
                    ctx, "subgoal_replan_needed",
                    f"子目标 #{sg['id']} 重做 {_tries} 次仍失败"
                    + ("（失败指向规划/依赖前提）" if _plan_err else "")
                    + " → 建议重规划后续子目标")
                try:
                    ctx.revise_feedback = list(
                        getattr(ctx, "revise_feedback", []) or []) + [
                        f"子目标 #{sg['id']}「{sg['title']}」反复失败"
                        f"（{str(step_result)[:80]}）：该子目标的规划粒度或"
                        "依赖前提可能有误，请重规划后续子目标后重做"]
                except Exception:  # noqa: BLE001
                    pass
    return step_result


def sg_tool_exec(self, ctx, sg, step_result, subgoal_plan_summary,
                 prev_results) -> str:
    """按蓝图声明的 `tool` 跑一次机器核验；未过则**带反馈重解一次**（2026-10-15）。

    与 `sg_light_check_resolve`（0-LLM **通用**校验）同构，但判据来自**蓝图声明**
    而非通用规则：声明 lean 就真去编译、声明 sympy 就真去判等式、声明 numeric
    就真去跑 norm_num。此前这些工具只有"模型恰好写了断言时被动触发"一条路径，
    而 lean 在子目标粒度上**根本没有执行入口** ⇒ 声明等于空文。

    ★ 重解**只有一次**（与 S1-lite 同口径）：仍不过就保留首解并留痕 `fail`，
      绝不无限循环，也绝不因机器核验失败就把结果丢弃（本项目铁律：校验失败
      要带反馈修正，不是把对的改错）。
    """
    if not step_result or step_result.startswith("[子目标"):
        return step_result
    _tool = str((sg or {}).get("tool") or "").strip().lower()
    if not _tool or _tool == "llm":
        return step_result          # 默认叙述路径，无机器核验
    _fb = self._tool_declared_verify(ctx, sg, step_result)
    if not _fb:
        return step_result
    self.record(ctx, "subgoal_tool",
                f"子目标 #{sg['id']}「{sg['title']}」工具核验未过"
                f"（tool={_tool}）: {_fb[:120]}，带反馈重解一次")
    _retry = self._solve_subgoal(ctx, sg, subgoal_plan_summary, prev_results,
                                 extra_hint=_fb)
    if _retry and not _retry.startswith("[子目标"):
        # 二次核验：过了就记 pass，仍不过保留重解结果并留痕 fail
        _fb2 = self._tool_declared_verify(ctx, sg, _retry)
        if not _fb2:
            self.record(ctx, "subgoal_tool",
                        f"子目标 #{sg['id']} 工具核验重解后通过（tool={_tool}）")
        return _retry
    return step_result


def solve_subgoals(self, ctx, subgoals, subgoal_plan_summary, results_map) -> int:
    """逐个求解子目标并落盘；返回累计前序注入字符数 `_ctx_inject_chars`。"""
    from .sub_goal_solver import _artifact_append
    _ctx_inject_chars = 0  # S4-lite 指标：累计前序注入字符数（近似上下文量）
    # 2026-09-12 新增（用户要求「原本完成的子目标要记录，不能重头再来」）：
    # 以**子目标指纹**为键，跨 run() 调用复用已完成结果。
    # 为何需要：orchestrator 会在 3_solve / 3.5 等阶段**多次**调用本 run()，
    # 原实现每次都重新规划并**重解全部子目标**（每个都是一次完整 LLM 轮），
    # 重复烧掉本就吃紧的单题预算（冒烟实测 700-1200s 的主要浪费源之一）。
    # 关闭 `subgoal_reuse_done` 即回到旧行为（每次全量重解）。
    _reuse_on = bool(getattr(self.config, "subgoal_reuse_done", True))
    _done = getattr(ctx, "_subgoal_done", None)
    if not isinstance(_done, dict):
        _done = {}
        try:
            setattr(ctx, "_subgoal_done", _done)
        except Exception:  # noqa: BLE001  个别 ctx 实现可能禁写
            pass
    _n_reuse = 0

    for sg in subgoals:
        # ★ 复用命中：上一轮已完成的**同一子目标** → 直接取结果，零 LLM 调用。
        # 指纹用 title+description+type（重新规划后 id 可能重排，语义内容不变）。
        _fp = "|".join(str(sg.get(_k, "")) for _k in
                       ("title", "description", "type"))[:240]
        # ★ 2026-10-20：缓存形态的规范化统一走 `subgoal_result.cached_*`，
        #   本路径不再自己判 isinstance；空缓存**不算命中**（否则会用缓存把
        #   一次真实求解吞掉，把空壳灌回两个容器）。写入走 `write_step()`。
        if _reuse_on and _fp in _done and cached_text(_done[_fp]):
            _cached = _done[_fp]
            step_result = cached_text(_cached)
            self._restore_contract(sg, cached_contract(_cached))
            write_step(sg, results_map, step_result)
            ctx.subgoal_trace.append(
                self._build_trace_item(sg, step_result, reused=True))
            # ★ 2026-10-02 hook 04：子目标结果即时落盘（存储层 04，append-only jsonl）。
            try:
                _artifact_append(ctx, "04_subgoal_results", ctx.subgoal_trace[-1])
            except Exception:  # noqa: BLE001
                pass
            _n_reuse += 1
            continue
        # 2026-09-17 **移除「因时间不足跳过剩余子目标」约束**
        # （用户明确要求：研究阶段不设时间预算、时间/token 无限，一切为正确率让路）。
        # 原两处闸（`if ctx.gen_time_up(): break` 与 `if _stage_left() <= 0: break`）
        # 会在生成侧软截止 / 子目标阶段预算耗尽时直接放弃剩余子目标 —— 后半段
        # 一旦被跳过，merge 只能拿残缺子目标合成（025 类题即此形态）⇒ 必错。
        # 循环本身由 `for sg in subgoals` 界定，删除后不存在无限循环风险；
        # stage_budget 仍写入 ctx._subgoal_stage_budget 供诊断，仅不再作为中断条件。

        # 2026-09-06 老师建议：子目标独立性 + 最小上下文依赖。
        # subgoal_ctx_mode = "deps"（默认）：按 depends_on 只注入直接依赖结果，
        # 无依赖子目标零前序上下文（可独立/并行，不被无关中间量污染）；
        # = "all"：回退旧行为全量前序注入（A/B 对照/兜底）。
        _ctx_mode = str(getattr(
            self.config, "subgoal_ctx_mode", "deps") or "deps").lower()
        if _ctx_mode == "all":
            prev_results = self._format_previous_results(
                results_map, subgoals)
        else:
            prev_results = self._format_dep_results(
                results_map, subgoals, sg)
        _ctx_inject_chars += len(prev_results or "")
        ctx.subgoal_ctx_inject_chars = _ctx_inject_chars  # 每步累计（预算中断也留痕）
        step_result = self._solve_subgoal(ctx, sg, subgoal_plan_summary, prev_results)
        # ★ 2026-10-13（用户 E24「让大模型反复写一道题」）：**子目标级无条件自改进**。
        #   「无条件」= 不看有无增益信号，固定跑 N 轮（与 3.3 主求解自改进同语义）；
        #   若某轮产出与上一轮相同/为空 ⇒ 提前停（省下无谓轮次，但**不因单轮失败放弃**）。
        _si_rounds = int(getattr(
            self.config, "subgoal_self_improve_rounds", 3) or 0)
        _si_done = 0
        if _si_rounds > 0 and step_result and not ctx.is_time_critical():
            for _si_r in range(_si_rounds):
                _si_new = self._improve_subgoal_result(ctx, sg, step_result)
                if not _si_new or _si_new.strip() == str(step_result).strip():
                    self.record(ctx, "subgoal_improve",
                                "子目标 %s 第 %d/%d 轮无新产出 ⇒ 停止"
                                % (sg.get("id"), _si_r + 1, _si_rounds))
                    break
                step_result = _si_new
                _si_done = _si_r + 1
            if _si_done:
                self.record(ctx, "subgoal_improve",
                            "子目标 %s 无条件自改进 %d 轮完成"
                            % (sg.get("id"), _si_done))
        # 2026-09-02 老师方案 B：蓝图评审 OK 但子目标失败 → 重做子目标
        # （不重画蓝图）。一次失败常是瞬时 LLM 错误/预算抖动，带已解
        # 子目标上下文重试一次；仍失败才记为占位（留给外层占位符兜底）。
        # 2026-09-17 移除重试的时间闸（原 `and _stage_left() > 120`）：时间无限，
        # 子目标失败一律带上下文重试一次；配套的「剩余不足则放弃重试」分支
        # （原 `elif ... _stage_left() <= 120`，仅写一条"放弃重试"记录）一并删除。
        # ★★ 2026-10-22（产出闸修复）：原触发条件只认 `startswith("[子目标")`
        #   —— 而实测 628 个子目标里 135 个失败（21.5%），其中 **95 个是
        #   `result == ""`（空返回）**：既不匹配占位符前缀，就**从不重试**，
        #   空结果直接流进 results_map 被 merge 当有效输入。
        #   空返回的成因多为"推理把 max_tokens 吃光后正文为空"（同 4.1 的
        #   `finish_reason=length` 家族），属于**可重做**的技术性失败。
        #   现在：空/纯空白 与 占位符 走同一条重做路径。
        if step_result.startswith("[子目标") or not str(step_result or "").strip():
            _why = ("（结果为空，疑似被截断或调用失败）"
                    if not str(step_result or "").strip() else "")
            self.record(ctx, "subgoal",
                        f"子目标 #{sg['id']}「{sg['title']}」失败{_why}，带上下文重试一次")
            if _ctx_mode == "all":
                prev_results2 = self._format_previous_results(
                    results_map, subgoals)
            else:
                prev_results2 = self._format_dep_results(
                    results_map, subgoals, sg)
            # ★ 2026-10-22：重做**多轮**（默认 2）—— 空返回多为推理被截断，
            #   换一次未必成功；且要带上"上一次为空"的提示（extra_hint），
            #   否则同一提示词大概率再空一次。
            _rr = max(1, int(getattr(
                self.config, "subgoal_failed_retry_rounds", 2) or 2))
            _empty = not str(step_result or "").strip()
            for _ri in range(_rr):
                retry = self._solve_subgoal(
                    ctx, sg, subgoal_plan_summary, prev_results2,
                    extra_hint=("上一次本步结果为空（可能被截断）。请重新给出完整结果，"
                                "必须包含【本步结果】【推导过程】，并以「本步结论: …」"
                                "开头给出可判定的结论，不要留空。"
                                if _empty else ""))
                if retry and not retry.startswith("[子目标"):
                    step_result = retry
                    break
                if retry:
                    step_result = retry
                self.record(ctx, "subgoal",
                            "子目标 #%s 第 %d/%d 次重做仍无有效结论"
                            % (sg.get("id"), _ri + 1, _rr))
            if not str(step_result or "").strip():
                # 重做仍空 ⇒ 落**显式占位符**（下游据此知道"本步无结论"，
                # 而不是拿到空串被当成"有内容但为空"）。
                step_result = "[子目标%s 求解失败]" % sg.get("id")

        step_result = sg_light_check_resolve(
            self, ctx, sg, step_result, subgoal_plan_summary, prev_results)
        # ★ 2026-10-15（用户要求「蓝图为每一个子目标设定工具，比如证明就用 lean」）：
        #   把蓝图声明的 tool 兑现成真正的机器核验。
        #   位置必须在 `_strip_check_tags` **之前** —— sympy/numeric 的判据就是
        #   `<check>` 断言，剥掉标签后就无从核验了。
        step_result = sg_tool_exec(
            self, ctx, sg, step_result, subgoal_plan_summary, prev_results)
        step_result = sg_adaptive_recover(
            self, ctx, sg, step_result, subgoal_plan_summary, prev_results)
        # ★ 2026-10-15（用户「2 与 3 之间建一个环」）：**求解期动态倒推**。
        #   位置刻意排在 `sg_adaptive_recover` **之后**：
        #     · recover = 不拆图的恢复（重做该子目标 / 交上游重规划），便宜，先跑；
        #     · 本步 = 拆图的恢复（现场倒推补一层前提再重解），贵（+1~2 次 LLM），
        #       只在"仍失败且失败指向缺前提"时才兜底。
        #   此前这种子目标只能写 `ctx.revise_feedback` 交上游，而上游 replan 已于
        #   2026-10-15 整块删除 ⇒ 反馈无人消费、子目标以占位符放行。本调用是那条
        #   断裂链的替代。实现与设计取舍见 sub_goal_solver__backward.py。
        step_result = sg_dynamic_backward(
            self, ctx, sg, step_result, subgoal_plan_summary, prev_results,
            results_map)
        # 2026-09-08：剥 <check> 验证标签（去标签留内容）——校验已在上面
        # 消费过 <check> 断言，存盘/注入下游（merge/lemma/后续子目标）时
        # 不得再带协议标记，防污染提示词与最终答案。
        step_result = self._strip_check_tags(step_result)
        # ★ 2026-10-20：唯一写入口（两容器同值），与协作路径共用。
        write_step(sg, results_map, step_result)
        if _reuse_on:
            # 写回 ctx，供**后续 run() 调用**复用（连同结论契约字段一起存，
            # 否则复用路径拿不到 has_derivation / 客观判定 → 误标"无推导"）。
            _done[_fp] = {"result": step_result,
                          "contract": self._contract_snapshot(sg)}
        # v2.9：结构化输出每步子目标的过程与中间结果
        ctx.subgoal_trace.append(self._build_trace_item(sg, step_result))
        # ★ 2026-10-02 hook 04：子目标结果即时落盘（存储层 04，append-only jsonl）。
        try:
            _artifact_append(ctx, "04_subgoal_results", ctx.subgoal_trace[-1])
        except Exception:  # noqa: BLE001
            pass

        # 引理积累（D6）：子目标求解成功后把结论写入 ctx.lemma_repo，
        # 供后续子目标与最终求解复用。
        # 2026-08-29 修复：此前 lemma_repo **全流水线无人写入**，
        # use_lemma_accumulation=True 等于读空列表（假钥匙，开了也是空转）。
        # 2026-08-29 二次修复：按领域路由（_use_lemma），数论开、其他关。
        if self._use_lemma(ctx):
            self._accumulate_lemma(ctx, sg, step_result)

        self.record(ctx, "subgoal_step",
                   f"子目标 #{sg['id']}「{sg['title']}」求解完成: {step_result[:80]}")
        time.sleep(0.2)  # 速率限制间隔
    return _ctx_inject_chars
