# Known failure patterns — MathPilot

Each entry is a failure mode that has been observed in real traces, the editable
unit it lives in, and the direction that has NOT yet been tried. A pattern is a
lead, not a solution: verify it on the trace before proposing an edit.

## 1. A correct answer gets evicted upstream

**Symptom**: `predicted` equals `gold`, yet `empty_answer` is true and the
formatter printed "no valid solution". The candidate existed and was correct.

**Where**: `agent/math_check_gate.py` — the calculation gate drops an entire
candidate when any extracted equation recomputes to a different value
(`n_bad > 0`, zero tolerance). A single slip in an intermediate step therefore
kills a correct conclusion.

**Direction not yet tried**: classify *which layer* the error sits in
(intermediate arithmetic vs final claim) before deciding admissibility. Raising
the tolerance ratio is not a fix — it lets real errors through in proportion and
cannot be justified.

**Risk**: any relaxation must still catch a wrong final answer.

## 2. Rewrite loop burns the budget without changing the answer

**Symptom**: `elapsed_sec` above 2400. The rewrite gate fired because the
candidate pool was empty, re-ran the whole chain, emptied again, and the
formatter gave up.

**Where**: `agent/rewrite_gate.py` (trigger) and `agent/orchestrator.py`.

**Direction not yet tried**: keep the best existing candidate as a low-confidence
fallback instead of re-running. This does not call the model again — it ranks
what is already there. Distinguish it from "guess an answer": the fallback is
still evidence-ranked.

**Risk**: masking a genuine failure behind a low-confidence answer. Mark it.

## 3. Truncation cuts the conclusion, not the middle

**Symptom**: a stage's reasoning is truncated and the tail — where the answer
lives — is gone.

**Where**: `agent/self_auditor.py` (`_MAX_TOKENS`, `_REASONING_MAX`) and similar
budgets in `answer_falsifier.py`, `verifier__deep.py`, `subgoal_solver`.

**Direction not yet tried**: keep-the-conclusion trimming (preserve the final
paragraph, compress the middle) instead of a plain head slice.

**Risk**: raising the budget instead is a cost regression; it must be paid for
by a measured gain.

## 4. Empty response treated as a wrong answer

**Symptom**: `error_class` mentions an empty or timed-out completion, and the
task scores 0 as if the model had answered wrongly.

**Where**: `utils/llm_client.py` (timeouts, retry), stage call sites.

**Direction not yet tried**: separate "the model said nothing" from "the model
said something wrong" in the per-stage verdicts, so the analyst can tell them
apart.

**Risk**: retries add cost; a retry that succeeds hides a stability problem.

## 5. Cost is structural, not per-token

**Symptom**: a 154-character problem costs as much as a 639-character one.

**Cause**: fixed stage count plus rewrite amplification, not input length.

**Implication for your edits**: an edit that shaves 10% off one stage saves
almost nothing. An edit that removes one rewrite saves more than every
micro-optimisation combined. Choose accordingly.
