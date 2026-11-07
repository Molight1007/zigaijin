"""答案抽取内核 + 救援兜底（自 utils/extract.py 逐字搬出）。

2026-10-08 批 D-ex1：`utils/extract.py` 超过 800 行上限，本模块承载其
**抽取/救援内核**（纯规则、仅依赖 `re`）：
  · 元语句判定 + 空壳/不完整判别（_META_PATTERNS … _has_answer_content）
  · 多级抽取 extract_final_answer + _extract_final_answer_impl
  · 工具调用 / calc 残留剥壳（_strip_toolcall_markers / _strip_calc_markers）
  · 续写占位符剥除 + clean_answer
  · 救援兜底（boxed / 强模式 / 尾行）+ 推理碎片/截断清洗（rescue_final_answer）

对外契约：`from utils.extract import <X>` 仍可用（主文件跳板式 re-export，同一对象 is）。
本模块**不导入** agent/tools（守 utils 分层 §2.4），也不反向导入主文件（无环）。
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------
# 元语句模式：模型推理的收尾/自我怀疑/格式化提示，不是答案
# ------------------------------------------------------------------
_META_PATTERNS = [
    # 收尾/整理语句
    r"(?:现在|接下来|然后|最后|最终)\s*[,，]?\s*(?:将|请|可以|需要)\s*.{0,20}(?:整理|结构化|格式化|呈现|表达|写出|总结|组织)",
    r"(?:将|请|可以|需要)\s*(?:上述|这些|以上|以上内容|这些步骤|上述步骤)\s*.{0,10}(?:整理|结构化|总结|写成|转化为|表示为)",
    # 自我怀疑/不确定/转折
    r"^或[者许]?[,，]?\s*可能.{0,30}(?:错[了误]|不对|有问题)",
    r"^(?:我(?:可能|好像|似乎|不确定|不太确定|怀疑)|可能我|或许我)",
    r"(?:我(?:需要|应该|还得|还要|再)|让我(?:们?))(?:重新.{0,10}(?:检查|验证|思考|计算|推导)|再.{0,10}(?:想|算|检查|看|确认))",
    r"^(?:等[等下下]|稍等|嗯+|唔+|呃+)[,，]?\s*",
    r"^(?:不过|但是|然而|可是|只是|然而)[,，]?\s*.{0,40}(?:不是|可能|也许|应该|需要|应该|想|考虑)",
    r"(?:这(?:可能|也许|应该|似乎)\s*.{0,20}(?:不是|不对|有误|有问题)|(?:不|没有).{0,5}(?:效|完整|确定|充分))",
    # 追问/反问（不含答案）
    r"^(?:为什么|怎么|如何|怎么样|是什么|这样可以吗|对吗|是吗|正确吗)",
    r"[?？]\s*$",  # 以问号结尾（通常在提问，不是给出答案）
    # 格式化占位/空答案前缀
    r"^(?:\[|\（|【).{0,5}(?:答案|最终答案|结果|解答|请在此|此处).{0,5}(?:\]|\）|】)",
    r"^(?:答案|结果|解答|最终答案|通解|特解|解集)[:：]?\s*$",
    r"^(?:通解|特解|解集|一般解)[:：]\s*$",
    # 纯结构/格式行
    r"^\s*(?:步骤\s*\d+|第\s*\d+\s*步|Step\s*\d+)\s*[：:：]*\s*$",
    # 被截断的 LaTeX 环境开头
    r"^\\begin\{[^}]+\}\s*$",
    r"^\\begin\{[^}]+\}\s*\\[a-zA-Z]+\s*$",
]

_META_RE = re.compile("|".join(_META_PATTERNS))


def _is_meta_line(line: str) -> bool:
    """判断一行文本是否是元语句（非答案内容）"""
    stripped = line.strip()
    if not stripped:
        return True
    # 纯标点
    if re.fullmatch(r"[\s,，。.、；;：:！!？?…\-—=_]+", stripped):
        return True
    # 匹配已知元模式
    if _META_RE.search(stripped):
        return True
    # 没有数学符号也没有数字/字母 → 大概率是叙述，不是答案
    has_math = bool(re.search(r"[$\\=<>≤≥→⇒]|\b[a-zA-Z]{2,}\b|[α-ωΑ-Ω]", stripped))
    has_number = bool(re.search(r"\d", stripped))
    has_answer_kw = bool(re.search(r"(?:[选选]项\s*[A-D]|正确[的选项]|答案[是为]|故选|故[选为])", stripped))
    # 超过50字符且没有数学符号/数字 → 极可能是叙述
    if len(stripped) > 50 and not has_math and not has_number and not has_answer_kw:
        return True
    return False


def _is_incomplete_answer(text: str) -> bool:
    """判断一个候选答案是否明显不完整，应被拒绝"""
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    # 以这些词结尾 → 只是前缀，不是完整答案
    if re.search(r"(?:通解|特解|解集|答案|最终答案|结果|应选|故)[为是:]\s*$", stripped, re.IGNORECASE):
        return True
    # 以 \begin{xxx} 结尾 → LaTeX 被截断
    if re.search(r"\\begin\{[^}]+\}\s*$", stripped):
        return True
    # 纯数学公式但非常短（< 10 字符）且没有答案标识 → 可能只是中间步骤
    if len(stripped) < 10 and not re.search(r"(?:=|答案|为|选)", stripped):
        return False  # 短公式不一定错，保留
    # 包含"但是""不过"等转折词且没有等号/答案标识 → 元叙述
    if re.search(r"(?:不过|但是|然而|可是|只是).{0,30}$", stripped) and not re.search(r"[=$＝]", stripped):
        return True
    return False


# 只由定界符 / 装饰符号 / 标题标记构成的"空壳"
# （实测失败样本：`$$`、`\[`、`## 最终答案`——模型答案区没写出内容）
_SHELL_ONLY = re.compile(
    r"^[\s\$\\!\[\]\(\)\{\}\.,;：:、，。\*\-\—_#`\"'<>|/+=~^]*$")
# 标题标记（Markdown 标题、"最终答案"字样等）本身不是答案
_TITLE_ONLY = re.compile(r"^#*\s*(最终答案|答案|解答|解|Answer|ANSWER)[：:#\s]*$", re.I)
# 可构成答案的实质内容：数字、字母、希腊字母、或常见数学/LaTeX 记号
_ANSWER_CONTENT = re.compile(
    r"[\d]|[A-Za-z]|[α-ωΑ-Ω]|[+\-*/^√∫∑∏∈⊂∪∩≠≤≥±∞→⇒⇔]")


def _has_answer_content(text: str) -> bool:
    r"""判断候选文本是否真的含有答案内容。

    用于拦截"空壳"输出：模型答案区只写了定界符（`$$`、`\[`）或标题
    （`## 最终答案`）时，此前会被兜底策略原样返回，最终作为答案提交给
    判分器——历史 45 条里有 4 条属于此类，是纯工程可修的丢分。
    """
    if not text:
        return False
    s = text.strip()
    if not s:
        return False
    if _SHELL_ONLY.match(s):
        return False
    # 去掉 Markdown 标题记号后若什么都不剩 → 空壳
    if _TITLE_ONLY.match(s):
        return False
    stripped_title = re.sub(r"^#+\s*", "", s)
    stripped_title = re.sub(r"^(最终答案|答案|解答|解|Answer|ANSWER)\s*[：:]?\s*",
                            "", stripped_title, flags=re.I).strip()
    if not stripped_title or _SHELL_ONLY.match(stripped_title):
        return False
    return bool(_ANSWER_CONTENT.search(s))


def _extract_last_valid_answer(text: str) -> str:
    """从文本尾部向前找第一个非元语句的有效行"""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    # 策略 A：从后往前找第一个包含 LaTeX/数学符号/选项标识的非元行
    for i in range(len(lines) - 1, max(len(lines) - 20, -1), -1):
        line = lines[i]
        if _is_meta_line(line):
            continue
        # 高优先级：明确答案标识
        if re.search(r"(?:[选选]项\s*[A-D]|正确[的选项]|故[选为]|答案为?|最终答案|答案[是为]|应选|综上[,，]?\s*[A-D])", line):
            return clean_answer(line)
    # 策略 B：找第一个含数学内容的非元行
    for i in range(len(lines) - 1, max(len(lines) - 20, -1), -1):
        line = lines[i]
        if _is_meta_line(line):
            continue
        if re.search(r"[$\\=]|\d", line):
            return clean_answer(line)
    # 策略 C：最后一道防线 — 找第一个非元行
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if not _is_meta_line(line):
            return clean_answer(line)
    return ""


def extract_final_answer(text: str) -> str:
    r"""从模型输出中提取最终答案（对外入口）。

    在内部多级策略的结果之上加一道出口校验：若抽出的是"空壳"
    （`$$` / `\[` / `## 最终答案` 等不含实质答案内容的字符串），返回空串，
    让调用方（solver.py:200、formatter 的 fallback）有机会换用其它候选，
    而不是把定界符当答案提交给判分器。
    历史 45 条里 4 条属于此类，是纯工程可修的丢分。
    2026-09-06 P4：出口增加 calc 工具标记剥除——模型截断在 `<calc>` 处时
    产物是裸标记（comb-022 实测 pred='<calc>' 整题格式丢分），resolve 正则需要
    闭合标签所以不处理，这里统一剥除后按空壳走调用方兜底。

    ★★ 2026-09-17 修复（**剥壳顺序错误**，实测捕获）：原先的写法是
        ans = _extract_final_answer_impl(text)   # ← 在**未剥壳**的原文上抽
        if ans:
            ans = _strip_calc_markers(ans)
            ans = _strip_toolcall_markers(ans)
    ⇒ **只剥"抽出来的结果"，不剥"抽取用的原文"**。
    实测 `<parameter=query>\\nsubset sum …1810\\n</parameter>`：
    `_extract_final_answer_impl` 先在第 3 步「行兜底」里把**参数内容**
    （`subset sum …1810`）当成"含数学内容的有效行"抽出来，
    然后 `_strip_toolcall_markers` 作用在**已抽出的纯文本**上——里面早就没有标签了
    ⇒ 剥了个寂寞，参数内容照样作为答案返回。
    （013 的候选 id=0 就是这个形态：答案 = 搜索 query 原文。）

    ⇒ 改为**先剥原文、再抽取**。这样第 3 步行兜底看到的原文里已无标签，
    整段纯标签输入直接变成空串，`impl` 无从抽取。
    保留出口的二次剥壳（幂等，防止 impl 内部自行拼出带标签的结果）。
    """
    # ★ 第一步：先对**原文**剥壳（顺序是关键，见上）
    text = _strip_toolcall_markers(_strip_calc_markers(text)) if text else text
    if not text or not text.strip():
        return ""
    ans = _extract_final_answer_impl(text)
    if ans:
        ans = _strip_calc_markers(ans)
        # ★ 2026-09-17：再剥**工具调用 XML 残留**（`<parameter=query>…</parameter>` 等）。
        #   实测 013：模型把 web_search 调用写成 XML 风格文本（非标准 tool_calls
        #   JSON），工具层不识别 ⇒ 原样留在响应里 ⇒ 被当成了"候选答案"。
        ans = _strip_toolcall_markers(ans)
    if ans and not _has_answer_content(ans):
        return ""
    return ans


_CALC_BLOCK_RE = re.compile(r"<calc\b[^>]*>.*?</calc>", re.DOTALL | re.IGNORECASE)
_CALC_STRAG_RE = re.compile(r"</?calc\b[^>]*>?", re.IGNORECASE)

# ★★ 2026-09-17 新增：**工具调用 XML 残留**剔除。
#   实测 official112-013（E 轮）：候选答案里出现了
#     `<parameter=query>`           ← 工具调用的参数标签
#     `subset sum逼近 1810 0≤x_i≤100 minimal d 竞赛题`  ← 该参数的内容
#     `看起来 d = 10 是候选答案。让我验证是否 d = 10 足够。` ← 推测语气
#   根因：模型有时**不用标准 `tool_calls` JSON**，而是把工具调用写成
#   **XML 风格的文本**（`<parameter=query>…</parameter>`）。而
#   `agent/base.py::llm_with_tools` **只处理 `resp["tool_calls"]`**（标准格式）
#   ⇒ 这段 XML 既没被执行、也没被剥除，**原样留在响应文本里**，
#   随后被答案抽取当成"答案"。003 的 `<parameter=que` 同源。
#   修法：与 `<calc>` 同样在答案抽取前剥除（完整块 + 裸标签残留）。
# ★★ 2026-09-17 修复（回归测试实测捕获）：原模式的 `tool_call` 后跟 `\b`，
#   而 `<tool_calls>` 里 `tool_call` 后面是 `s`（单词字符）⇒ `\b` **不成立**
#   ⇒ **复数形式 `<tool_calls>` / `</tool_calls>` 完全没被匹配**，
#   原样返回成答案（新写的 `test_no_path_returns_bare_tag` 当场抓到）。
#   ⇒ 改为 `tool_calls?`（可选复数），并把 `function` 一并允许复数。
_TOOLCALL_BLOCK_RE = re.compile(
    r"<(?:parameter|function|tool_calls?)\b[^>]*>.*?</(?:parameter|function|tool_calls?)>",
    re.DOTALL | re.IGNORECASE)
_TOOLCALL_OPEN_RE = re.compile(
    r"<(?:parameter|function|tool_calls?)\b[^>]*>?(?!.*?</(?:parameter|function|tool_calls?)>)",
    re.DOTALL | re.IGNORECASE)
_TOOLCALL_STRAG_RE = re.compile(
    r"</?(?:parameter|function|tool_calls?|invoke|antml:parameter|"
    r"antml:function_calls|antml:invoke)\b[^>]*>?",
    re.IGNORECASE)


def _strip_toolcall_markers(text: str) -> str:
    """剥除工具调用的 XML 残留（完整块优先，再处理裸标签及其参数内容）。

    ⚠ 三步顺序重要：
      ① 先删**成对块**；
      ② 再处理**裸开标签** —— 实测泄漏形态是 `<parameter=query>参数值`
         且**没有闭合标签**，仅删标签会留下参数值（`subset sum逼近 1810…`）
         照样被当成答案 ⇒ **必须连同其后的参数内容一起截断**；
      ③ 最后清残留的孤立标签。
    """
    if not text:
        return text
    text = _TOOLCALL_BLOCK_RE.sub("", text)
    _m = _TOOLCALL_OPEN_RE.search(text)
    if _m:
        text = text[:_m.start()]      # 裸开标签之后全属工具参数 ⇒ 截断
    return _TOOLCALL_STRAG_RE.sub("", text).strip()


def _strip_calc_markers(text: str) -> str:
    """剥除答案中的 calc 工具标记（完整块与裸 <calc/</calc> 残留）。"""
    if not text:
        return text
    text = _CALC_BLOCK_RE.sub("", text)
    return _CALC_STRAG_RE.sub("", text).strip()


def _extract_final_answer_impl(text: str) -> str:
    """
    从模型输出中提取最终答案。

    多级提取策略：
    1. 查找【最终答案】/ \\boxed{} 等显式标记
    2. 从文本尾部跳过元语句向前扫描
    3. LaTeX 表达式兜底
    4. 数字表达式兜底
    5. 全文兜底
    """
    if not text:
        return ""

    # 策略 1：【最终答案】/ 答案: / Final answer: 等显式标记
    # BUG-10 修复：多行答案用贪婪匹配到明确的节标记或文本末尾
    # P0-4 增强：兼容 prefill 答案前置形态（"最终答案：42" 后紧跟解释，
    # 答案在首行——优先取首行短答案，避免吞入后续推理）
    patterns = [
        (r"【最终答案】\s*\n?\s*([\s\S]+)", False),
        (r"##\s*最终答案\s*\n?\s*([\s\S]+?)(?=\n\s*##|$)", False),
        (r"##\s*[Ff]inal\s+[Aa]nswer\s*\n?\s*([\s\S]+?)(?=\n\s*##|$)", False),
        (r"最终答案[:：]\s*([\s\S]+)", False),
        (r"答案[:：]\s*([\s\S]+)", False),
        (r"[Ff]inal\s*[Aa]nswer[:：]\s*([\s\S]+)", False),
    ]
    for pattern, is_boxed in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            captured = match.group(1).strip()
            # prefill 答案前置：标记后首行即答案，优先返回短首行
            # P0-4 修复：剥离标记后遗留的 ": " 前缀（"【最终答案】: 5" 形态）
            first_line = re.sub(r"^[:：,，\s]+", "", captured.split("\n")[0].strip())
            if (first_line and first_line != captured
                    and 1 < len(first_line) <= 200
                    and re.search(r"\d|[a-zA-Zα-ω]|[$\\=＝]", first_line)
                    and not _is_meta_line(first_line)
                    and not _is_incomplete_answer(first_line)):
                return clean_answer(first_line)
            # 裁剪到第一个节标记（【...】/ #### / --- / == ）
            # P0-4 修复：允许单字符答案（prefill 场景 "最终答案：7" 是合法答案）
            captured = re.split(r'\n\n(?:【|####|---|==|答案|最终答案)', captured)[0].strip()
            if (captured and len(captured) >= 1
                    and not _is_meta_line(captured)
                    and not _is_incomplete_answer(captured)):
                return clean_answer(captured)

    # 策略 2：\\boxed{...} 格式（支持任意嵌套深度）
    ans = _extract_boxed_nested(text)
    if ans and not _is_incomplete_answer(ans):
        return ans

    # 策略 3：从尾部跳过元语句向前找有效答案行（核心修复）
    # ★★ 2026-10-07（单字符闸门残留 · A）：原判据 `len(last_valid) > 1` 与本文件
    #   已修口径**不一致** —— `is_valid_final_answer` 已删 `len(s)<2` 闸门
    #   （单字符是合法且完整的答案），策略 1 也用 `>= 1`，formatter 亦明述同口径。
    #   实测：策略 6 兜底仍能捞回裸单字符，故此处**未产生实际误杀**；但为消除
    #   口径分叉（选项字母 A/B/C/D、个位数答案的潜在误杀面），统一改为 `>= 1`。
    last_valid = _extract_last_valid_answer(text)
    if last_valid and len(last_valid) >= 1 and not _is_incomplete_answer(last_valid):
        return last_valid

    # 策略 4：扫描全文找 LaTeX $...$ / $$...$$ / \(...\) 取最后一个
    latex_match = re.findall(
        r"(?:\$\$?\s*)(.+?)(?:\s*\$?\$)|(?:\$)([^\$]+)(?:\$)|(?:\\\()(.+?)(?:\\\))",
        text,
    )
    if latex_match:
        candidates = []
        for groups in latex_match:
            for g in groups:
                if g and g.strip() and not _is_meta_line(g.strip()) and not _is_incomplete_answer(g.strip()):
                    candidates.append(g.strip())
        if candidates:
            return clean_answer(candidates[-1])

    # 策略 5：找最后一个数字/表达式结果
    num_match = re.findall(
        r"(?:[=＝]|结果是?|得到|求得|解得|答案为?)\s*"
        r"([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?"
        r"(?:\s*[-+*/×÷]\s*[-+]?\d*\.?\d+)*"
        r"(?:\s*[πeabeixyABXY𝜋]\^?\d*)*)",
        text,
    )
    if num_match:
        last = num_match[-1].strip()
        if not _is_meta_line(last) and not _is_incomplete_answer(last):
            return last

    # 策略 6：尝试忽略元语句后重新取倒数几行
    if lines := [l.strip() for l in text.strip().split("\n") if l.strip()]:
        clean_lines = [
            l for l in lines
            if not _is_meta_line(l)
            and not _is_incomplete_answer(l)
            and _has_answer_content(l)   # 过滤 `$$` / `\[` / `## 最终答案` 这类空壳
        ]
        if clean_lines:
            return clean_answer(clean_lines[-1])

    # 兜底：返回全文（确保验证器有内容可评估）
    # 但若全文本身不含任何答案内容（纯定界符 / 只有标题），返回空串——
    # 让调用方（solver.py:200、formatter 的 fallback）有机会换用其它候选，
    # 而不是把 `$$` 这种垃圾当成答案直接提交给判分器。
    stripped_all = text.strip()
    return stripped_all if _has_answer_content(stripped_all) else ""


# 模型自产"续写占位符"：Intern 系列在收尾时会回显 `[续写]` / `--- 请继续 ---`
# 之类的交接标记（2026-08-29 lemma A/B 实测：algebra-075 答案
# `3\n\n[续写]\n--- 请继续 ---` 因混入该标记被严格判分器判错）。
_CONTINUATION_MARKERS = [
    r"\[\s*续写\s*\]", r"-{2,}\s*请继续\s*-{2,}",
    r"请继续(?:完成|推理)?", r"\[TBC\]", r"to\s*be\s*continued",
]


def _strip_continuation_markers(text: str) -> str:
    """剔除答案文本里的续写占位符（纯防御，不改变正常内容）。"""
    if not text:
        return text
    out = text
    for pat in _CONTINUATION_MARKERS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    # 剔除后收拢多余空白（`x = 2 [续写] y = 3` → `x = 2  y = 3` → 单空格）
    out = re.sub(r" {2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def clean_answer(text: str) -> str:
    """清理答案文本，去除多余符号"""
    text = _strip_continuation_markers(text).strip()
    # 去除编号前缀
    text = re.sub(r"^[\d]+[\.\、\)）]\s*", "", text)
    # 去除 markdown 格式
    text = text.replace("**", "").replace("__", "")
    return text


# ============================================================
# 兜底提取增强（同步自 测试工具/intern_s1.py，纯规则、不耗 LLM 预算）
# ============================================================


def _extract_boxed_nested(text: str) -> str:
    """匹配 \\boxed{...}，通过手动计数大括号处理任意嵌套深度。"""
    if not text:
        return ""

    marker = r"\boxed{"
    results = []
    idx = 0
    while True:
        pos = text.find(marker, idx)
        if pos == -1:
            break
        start = pos + len(marker)  # 内容起始位置（{ 之后）
        depth = 1
        cursor = start
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    results.append(text[start:cursor].strip())
                    break
            cursor += 1
        idx = pos + 1

    return results[-1] if results else ""


# 强模式：明确结论句（可靠，可在全文搜索）
_ANSWER_PATTERNS = [
    # "the (final) answer is X" / "answer: X" — 最可靠
    (r"(?i)(?:the\s+(?:final\s+|correct\s+)?answer\s+(?:is\s*[:\s]|:))\s*(.+?)(?:\.\s*$|\.\s|\n\n|\Z)", 1),
    # "X = Y is a candidate / solution / the answer"（Intern-S1 常用）
    (r"(?i)\b([a-zA-Z])\s*=\s*(-?\d+)\s+is\s+(?:a\s+(?:valid\s+)?(?:candidate|solution)|the\s+answer)", 2),
    # "Therefore/Thus/Hence/So, X." — 需要 X 是短答案
    (r"(?i)(?:Therefore|Thus|Hence|So)\s*[,:]\s*(.{1,120}?)(?:\.\s*$|\n\n|\Z)", 1),
    # "which gives X" / "which yields X" / "we conclude X"
    (r"(?i)(?:which\s+(?:gives|yields)|we\s+conclude)\s+(.{1,120}?)(?:\.\s*$|\n\n|\Z)", 1),
    # "result: X" / "conclusion: X"
    (r"(?i)(?:(?:final\s+)?result|conclusion)\s*:\s*(.{1,120}?)(?:\.\s*$|\n\n|\Z)", 1),
]

# 弱模式：行尾变量赋值（如 "N = 3"），只在文本尾部搜索，避免误匹配推理开头
_WEAK_ASSIGN_PATTERN = (r"(?m)^(?:\w)\s*=\s*(.{1,80}?)\s*$", 1)


def _extract_strong_pattern(text: str) -> str:
    """仅使用强结论模式匹配（可靠）：
    1) 强模式全文搜索（候选解如 "N=3 is a candidate" 常出现在推理中段）
    2) 弱赋值模式只在尾部搜索（结论通常在末尾，避免开头误匹配）
    """
    if not text or len(text) < 10:
        return ""

    tail = text[-4000:] if len(text) > 4000 else text

    # 1) 强模式在全文搜索
    for pattern, group in _ANSWER_PATTERNS:
        m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if m:
            candidate = m.group(group).strip()
            candidate = _clean_extracted_answer(candidate)
            if candidate and 1 <= len(candidate) <= 200:
                return candidate

    # 2) 弱赋值模式只在尾部搜索
    m = re.search(_WEAK_ASSIGN_PATTERN[0], tail, re.MULTILINE)
    if m:
        candidate = m.group(_WEAK_ASSIGN_PATTERN[1]).strip()
        candidate = _clean_extracted_answer(candidate)
        if candidate and 1 <= len(candidate) <= 200:
            return candidate

    return ""


def _extract_tail_fallback(text: str) -> str:
    """最后手段：取最后一行非空内容作为答案（误报率高，仅作兜底）"""
    if not text:
        return ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in reversed(lines[-5:]):  # 只看最后5行
        # 跳过推理标记、半截句子
        skip_words = ["```", "wait", "let me", "now consider", "for example",
                      "we can", "since", "because", "first", "next", "then",
                      "note that", "assume", "suppose", "let's"]
        if any(line.lower().startswith(w) for w in skip_words):
            continue
        # 跳过以 "(" "[" "{" 开头（通常是公式推导）
        if line and line[0] in "([{":
            continue
        cleaned = _clean_extracted_answer(line)
        # 推理片段检测（清理后的内容仍像推理句则跳过）
        if cleaned and not _looks_like_reasoning_fragment(cleaned) and 3 <= len(cleaned) <= 200:
            return cleaned

    return ""


_REASONING_FRAGMENT_WORDS = [
    "if we", "we set", "let us", "we have", "we can", "we get", "we need",
    "so best", "it is decreasing", "note that", "assume", "suppose",
    "since", "because", "first", "next", "then", "wait", "for each",
    "for example", "the number written is", "the set", "term is",
]


def _has_unbalanced_math_delims(text: str) -> bool:
    """LaTeX 数学定界符配对检查（2026-09-11）。

    #006 实况（Bug清单_团队提交de90cedc_0911.md §6）：答案抽取在非数值题上
    退化成**截断的推理正文**——`假设 \\( A(0) = 0 \\)，代入 \\( p = 1 \\) 和 \\( q`
    （39 字符，句子中间硬断）。这类串同时造成两种危害：
      ① 提交内容本身不可判分（必错）；
      ② 被当作锚串喂给 Lean，污染 `_answer_embedded` 判定。

    只做**成对**检查：任一成对定界符数量为奇数（或 `$` 为奇数）即判不平衡。
    两侧都不出现时（纯文本/纯表达式答案）不算不平衡，不影响既有通过率。
    返回 True 表示存在未闭合定界符 → 应视为无效答案。
    """
    if not text:
        return False
    if len(re.findall(r"\\\(", text)) != len(re.findall(r"\\\)", text)):
        return True
    if len(re.findall(r"\\\[", text)) != len(re.findall(r"\\\]", text)):
        return True
    # 行内 $（排除转义 \$）
    if len(re.findall(r"(?<!\\)\$", text)) % 2 != 0:
        return True
    return False


def _clean_extracted_answer(text: str) -> str:
    """清理提取的答案文本；若清理后仍呈推理片段特征则返回空串"""
    if not text:
        return ""
    # 去掉开头的冒号、空格、破折号
    text = re.sub(r"^[:\s\-–—]+", "", text)
    # 去掉尾部的空格和标点
    text = text.strip().rstrip(".;,，。；")
    # 去掉开头的小写引导词
    text = re.sub(r"^(?:is|that|it\s+is|the\s+answer\s+is)\s+", "", text, flags=re.IGNORECASE)
    # 去掉编号前缀（如 "1. If we set..."）
    # 注意：必须是编号分隔符后紧跟空白或结尾才删除，避免误删 "3.5" 的小数点
    text = re.sub(r"^\d+[.)](?=\s|$)", "", text).strip()
    # 推理片段特征检测：以引导词开头 → 非独立答案
    if _looks_like_reasoning_fragment(text):
        return ""
    # 结构完整性检测：LaTeX 定界符未配对 = 截断的推理正文（#006 实况）
    if _has_unbalanced_math_delims(text):
        return ""
    # 保留下划线、反斜杠、花括号等数学符号
    return text.strip()


def _looks_like_reasoning_fragment(text: str) -> bool:
    """判断文本是否像推理片段而非独立答案"""
    if not text:
        return False
    low = text.lower()
    for w in _REASONING_FRAGMENT_WORDS:
        if low.startswith(w):
            return True
    # 以 "X=..." 开头但后面跟着推理连词（如 "2: π(2)=3 ≤2, so floor=1"）
    if re.match(r"^\w+\s*[:=]", low) and re.search(r"\b(so|then|since|because|thus)\b", low):
        return True
    return False


def rescue_final_answer(text: str) -> tuple[str, str]:
    """答案兜底提取（纯规则、不消耗 LLM 预算）：依次尝试
    嵌套 boxed → 强结论模式 → 尾部兜底。

    同步自 测试工具/intern_s1.py::_rescue_answer，但去掉了 DeepSeek 跨模型
    通道（赛事提交版只有平台注入的单一 client，竞赛禁止硬编码 API Key）。

    ★★ 2026-09-17 修复（official112-013 实况）：本函数原先**没剥工具调用 XML**，
    而 `extract_final_answer` 只在自身出口剥 ⇒ 当响应是**纯工具调用残留**
    （`<tool_call>…</tool_call>` 或裸 `</tool_call>`）时：
      · `extract_final_answer` 正确返回空串（已剥壳）；
      · 但调用方随即回退到本函数，本函数**不剥** ⇒ 三个子策略全部命中原样标签，
        行兜底直接返回 `'</tool_call>'` 当成答案落盘。
    实测 013：6 个候选里 **4 个 `answer=='</tool_call>'`**、其完整推导（2683 字）
    留在 `reasoning` 里被丢弃 ⇒ **池里只剩垃圾 + 一个错值**，直接导致该题判错。
    ⇒ 在两处入口（本函数 + `smart_fallback_answer`）统一先剥壳，与
    `extract_final_answer` 口径对齐。剥壳后再判空壳，避免把标签当答案。

    参数:
        text: LLM 原始输出

    返回:
        (提取到的答案, 提取来源)；未提取到返回 ("", "")。
    """
    if not text:
        return "", ""
    # ★ 先剥工具调用残留 + calc 标记，否则 `</tool_call>` 会被当答案返回
    text = _strip_toolcall_markers(_strip_calc_markers(text)).strip()
    if not text:
        return "", ""

    # 1) 嵌套 boxed 提取
    answer = _extract_boxed_nested(text)
    if answer:
        cleaned = _clean_extracted_answer(answer)
        if cleaned:
            return cleaned, "boxed"

    # 2) 强结论模式（可在全文截获中段结论）
    answer = _extract_strong_pattern(text)
    if answer:
        return answer, "strong_pattern"

    # 3) 尾部兜底
    answer = _extract_tail_fallback(text)
    if answer:
        return answer, "tail_fallback"

    return "", ""
