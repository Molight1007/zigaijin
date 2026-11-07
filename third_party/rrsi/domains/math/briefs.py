"""Domain paragraphs for the three search roles on the MathPilot task bank."""

ANALYST = """The agent is a math-solving pipeline driven by a frozen policy LLM
({policy}). It answers competition-style problems through fixed stages:
question understanding -> blueprint DAG -> subgoal solving -> candidate
generation -> a five-link audit chain (self-audit / adversarial falsification /
Lean check / numeric recomputation / voting) -> final formatter. A task counts
as solved only if the final answer matches the gold answer after symbolic
normalisation (answers_match); there is no LLM judge, so the score is
deterministic per trial.

The dominant near-miss mode is NOT a wrong answer: it is a correct answer whose
candidate was discarded upstream. The calculation gate (stage 4.4) removes an
entire candidate when any extracted equation recomputes to a different value
(n_bad > 0, zero tolerance), so a single arithmetic slip in an intermediate
step kills a correct conclusion. The rewrite gate (stage 4.6) then empties the
candidate pool and the formatter prints "no valid solution". When a trace ends
in that string, read the audit-chain verdicts before concluding the model
cannot solve the problem: look for candidates that were correct and got evicted.

Cost is structural, not proportional to question length: one trial takes 15-35
minutes regardless of problem size, and a rewrite round roughly doubles it
because the whole chain re-runs. Prefer edits that remove a wasted rewrite over
edits that make a single stage marginally faster.
"""

DIGESTER = """A trial directory holds one JSON per stage:
01_understanding.json (question_type / domain / problem_goal / lean_applicable),
02_blueprint.json (the DAG), 03_subgoals.json, 04_subgoal_results.jsonl (one
line per solved subgoal), 05_candidates.json (a list of candidate answers, each
with id / answer / reasoning / revised), and the audit-chain verdicts inside the
result diag. Read 05_candidates.json first to see what the pipeline actually
produced, then the verdicts to see which gate removed what and why. A candidate
whose answer equals gold but whose verdict is "evicted" is the highest-value
signal in the whole trace.
"""

PROPOSER = """The benchmark is the MathPilot official-112 bank: competition-style
math problems, most of them answer-type (a closed-form expression or a number),
graded by symbolic comparison against a gold answer. The harness is a fixed
five-stage pipeline: question understanding, blueprint DAG, subgoal solving,
candidate generation, then a five-link audit chain (self-audit, adversarial
falsification, Lean verification, numeric recomputation, voting) and a final
formatter. The frozen policy is {policy}; do not assume it reasons the way you
do — read the trajectories for how it actually fails.

Out of bounds, enforced before measurement: any task identifier (for example
"official112-019" or a number-theory problem slug), any gold answer string, any
numeric or symbolic constant that only makes sense for one specific problem, and
any branch keyed on a problem's wording. Reading or special-casing the verifier
is likewise out. The edits are judged on held-out problems the search never
sees, so memorised problem knowledge is worthless.

Litmus test for every edit: would this help a competent human solver working on
MANY unfamiliar competition problems, or does it only help on the problem whose
trace you just read?
"""

CRITIC = """The harness is a five-stage math-solving pipeline graded by symbolic
answer comparison. General solving craft is fine ("before finalising, re-check
that every symbol introduced in the answer appears in the problem" = OK;
"expand a bound variable before substituting" = OK). Problem knowledge is not
("for the 2x2 block problem, Natasha does not interfere" = REJECT; quoting a
gold answer, a task id, or a numeric constant that only fits one problem =
REJECT).

Existing mechanisms that must not be disabled without a working replacement:
symbolic answer matching, the Lean verification link, and the adversarial
falsification pass. Removing a gate to raise the score is reward hacking, not an
improvement.
"""
