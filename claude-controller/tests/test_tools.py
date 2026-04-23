"""Unit tests for tools.py — queue recording, phase gating, and flow IDs."""

import asyncio
import unittest

from tools import (
    CandidatePool,
    DEFAULT_AUTONOMOUS_BUDGET,
    DecisionQueue,
    FindingFiled,
    PHASE_DIRECTION,
    PHASE_IDLE,
    PHASE_VERIFICATION,
    PlanEntry,
    WorkerDecision,
    _reject_wrong_phase,
    extract_flow_ids,
    reset_active_worker,
    set_active_worker,
)


class TestCandidatePool(unittest.TestCase):
    def test_contextvar_attribution(self):
        p = CandidatePool()
        token = set_active_worker(3)
        try:
            cid = p.add(title="T", severity="medium", endpoint="/z", flow_ids=["qq11rr"],
                        summary="s", evidence_notes="e", reproduction_hint="r")
        finally:
            reset_active_worker(token)
        self.assertEqual(p.get(cid).worker_id, 3)

    def test_contextvar_isolates_concurrent_tasks(self):
        """Each asyncio task sees its own active_worker_id."""
        p = CandidatePool()

        async def add_as(worker_id: int, title: str, delay: float) -> str:
            token = set_active_worker(worker_id)
            try:
                await asyncio.sleep(delay)  # interleave tasks
                return p.add(title=title, severity="low", endpoint="/",
                             flow_ids=["aaaa11"], summary="",
                             evidence_notes="", reproduction_hint="")
            finally:
                reset_active_worker(token)

        async def run_both():
            return await asyncio.gather(
                add_as(1, "from-1", 0.01),
                add_as(2, "from-2", 0.0),
            )

        cid_a, cid_b = asyncio.run(run_both())
        self.assertEqual(p.get(cid_a).worker_id, 1)
        self.assertEqual(p.get(cid_b).worker_id, 2)

    def test_pending_excludes_verified_and_dismissed(self):
        p = CandidatePool()
        c1 = p.add(title="A", severity="high", endpoint="/x", flow_ids=["a1b2c3"],
                   summary="", evidence_notes="", reproduction_hint="")
        c2 = p.add(title="B", severity="low", endpoint="/y", flow_ids=["d4e5f6"],
                   summary="", evidence_notes="", reproduction_hint="")
        c3 = p.add(title="C", severity="low", endpoint="/z", flow_ids=["g7h8i9"],
                   summary="", evidence_notes="", reproduction_hint="")
        p.mark(c1, "verified")
        p.mark(c3, "dismissed")
        self.assertEqual([c.candidate_id for c in p.pending()], [c2])

    def test_ids_since(self):
        p = CandidatePool()
        p.add(title="a", severity="low", endpoint="/", flow_ids=["aaaa11"],
              summary="", evidence_notes="", reproduction_hint="")
        before = p.counter
        p.add(title="b", severity="low", endpoint="/", flow_ids=["bbbb22"],
              summary="", evidence_notes="", reproduction_hint="")
        p.add(title="c", severity="low", endpoint="/", flow_ids=["cccc33"],
              summary="", evidence_notes="", reproduction_hint="")
        self.assertEqual(p.ids_since(before), ["c002", "c003"])

    def test_ids_since_for_worker_filters(self):
        p = CandidatePool()
        # Worker 1 adds one
        token = set_active_worker(1)
        try:
            p.add(title="w1", severity="low", endpoint="/", flow_ids=["aaaa11"],
                  summary="", evidence_notes="", reproduction_hint="")
        finally:
            reset_active_worker(token)
        before = p.counter
        # Worker 2 adds two
        token = set_active_worker(2)
        try:
            p.add(title="w2a", severity="low", endpoint="/", flow_ids=["bbbb22"],
                  summary="", evidence_notes="", reproduction_hint="")
            p.add(title="w2b", severity="low", endpoint="/", flow_ids=["cccc33"],
                  summary="", evidence_notes="", reproduction_hint="")
        finally:
            reset_active_worker(token)
        # Worker 1 adds another
        token = set_active_worker(1)
        try:
            p.add(title="w1b", severity="low", endpoint="/", flow_ids=["dddd44"],
                  summary="", evidence_notes="", reproduction_hint="")
        finally:
            reset_active_worker(token)

        self.assertEqual(p.ids_since_for_worker(before, 2), ["c002", "c003"])
        self.assertEqual(p.ids_since_for_worker(before, 1), ["c004"])
        self.assertEqual(p.ids_since_for_worker(before, 99), [])


class TestPlanAccumulation(unittest.TestCase):
    """Multiple plan_workers calls in one phase must accumulate.

    Claude routinely issues plan_workers across several tool calls
    (e.g. one call per new worker). If set_plan replaced rather than
    merged, only the last call's entries would survive, which manifests
    as "director wanted 3 workers but only 1 spawned".
    """

    def test_two_calls_different_ids_accumulate(self):
        q = DecisionQueue()
        q.begin_phase(PHASE_DIRECTION)
        q.set_plan([PlanEntry(2, "scan /api")])
        q.set_plan([PlanEntry(3, "scan /admin")])
        self.assertIsNotNone(q.plan)
        ids = [p.worker_id for p in q.plan]
        self.assertEqual(sorted(ids), [2, 3])

    def test_same_id_overrides_last_wins(self):
        q = DecisionQueue()
        q.begin_phase(PHASE_DIRECTION)
        q.set_plan([PlanEntry(2, "first")])
        q.set_plan([PlanEntry(2, "second")])
        self.assertEqual(len(q.plan), 1)
        self.assertEqual(q.plan[0].assignment, "second")

    def test_mixed_new_and_override(self):
        q = DecisionQueue()
        q.begin_phase(PHASE_DIRECTION)
        q.set_plan([PlanEntry(2, "a"), PlanEntry(3, "b")])
        q.set_plan([PlanEntry(3, "b2"), PlanEntry(4, "c")])
        by_id = {p.worker_id: p.assignment for p in q.plan}
        self.assertEqual(by_id, {2: "a", 3: "b2", 4: "c"})

    def test_reset_clears_accumulated_plan(self):
        q = DecisionQueue()
        q.begin_phase(PHASE_DIRECTION)
        q.set_plan([PlanEntry(2, "a"), PlanEntry(3, "b")])
        q.reset()
        self.assertIsNone(q.plan)

    def test_phase_boundary_does_not_carry_plan_across_iters(self):
        """After controller reset() between iterations, plans don't leak."""
        q = DecisionQueue()
        q.begin_phase(PHASE_DIRECTION)
        q.set_plan([PlanEntry(2, "iter-1")])
        q.reset()
        q.begin_phase(PHASE_DIRECTION)
        q.set_plan([PlanEntry(3, "iter-2")])
        self.assertEqual([p.worker_id for p in q.plan], [3])


class TestDecisionQueuePhases(unittest.TestCase):
    def test_reset_clears_all(self):
        q = DecisionQueue()
        q.begin_phase(PHASE_DIRECTION)
        q.set_plan([PlanEntry(1, "x")])
        q.add_decision(WorkerDecision(kind="continue", worker_id=1, instruction="i", progress="new"))
        q.begin_phase(PHASE_VERIFICATION)
        q.add_finding(FindingFiled(title="T", severity="high", endpoint="/", description="",
                                    reproduction_steps="", evidence="", impact="",
                                    verification_notes="v"))
        q.add_dismissal("c001", "false positive")
        q.set_verification_done("verified")
        q.begin_phase(PHASE_DIRECTION)
        q.set_direction_done("directed")
        q.set_done("wrap")

        q.reset()
        self.assertIsNone(q.plan)
        self.assertEqual(q.worker_decisions, [])
        self.assertEqual(q.findings, [])
        self.assertEqual(q.dismissals, [])
        self.assertIsNone(q.done_summary)
        self.assertIsNone(q.verification_done_summary)
        self.assertIsNone(q.direction_done_summary)
        self.assertEqual(q.phase, PHASE_IDLE)

    def test_begin_phase_transitions(self):
        q = DecisionQueue()
        self.assertEqual(q.current_phase, PHASE_IDLE)
        q.begin_phase(PHASE_VERIFICATION)
        self.assertEqual(q.current_phase, PHASE_VERIFICATION)
        q.set_verification_done("ok")
        self.assertEqual(q.verification_done_summary, "ok")
        # Re-entering a phase clears only its own done flag
        q.begin_phase(PHASE_VERIFICATION)
        self.assertIsNone(q.verification_done_summary)

    def test_begin_phase_does_not_clear_other_accumulators(self):
        q = DecisionQueue()
        q.begin_phase(PHASE_VERIFICATION)
        q.add_finding(FindingFiled(title="T", severity="high", endpoint="/", description="",
                                    reproduction_steps="", evidence="", impact="",
                                    verification_notes="v"))
        q.begin_phase(PHASE_DIRECTION)
        self.assertEqual(len(q.findings), 1)  # findings accumulate across phase switches within an iter


class TestRejectWrongPhase(unittest.TestCase):
    def test_error_shape_and_content(self):
        cases = [
            (PHASE_DIRECTION, PHASE_VERIFICATION, "plan_workers",
             ["plan_workers", "verification", "verification_done"]),
            (PHASE_VERIFICATION, PHASE_DIRECTION, "file_finding", ["file_finding"]),
        ]
        for current, expected, tool, must_include in cases:
            with self.subTest(tool=tool):
                out = _reject_wrong_phase(current, expected, tool)
                self.assertTrue(out["is_error"])
                for needle in must_include:
                    self.assertIn(needle, out["content"][0]["text"])


class TestWorkerDecisionDefaults(unittest.TestCase):
    def test_autonomous_budget_default(self):
        d = WorkerDecision(kind="continue", worker_id=1, instruction="go", progress="new")
        self.assertEqual(d.autonomous_budget, DEFAULT_AUTONOMOUS_BUDGET)


class TestExtractFlowIds(unittest.TestCase):
    def test_text_keyword_patterns(self):
        text = (
            "I opened flow_id=abcdef and also source_flow_id: DEF456. "
            'Nested: flow_a="xy12zz", flow_b=11qq2.'
        )
        ids = extract_flow_ids(text)
        for expected in ("abcdef", "DEF456", "xy12zz", "11qq2"):
            self.assertIn(expected, ids)

    def test_dict_flow_id_field(self):
        d = {"flow_id": "id0001", "inner": {"source_flow_id": "id0002"}}
        ids = extract_flow_ids(d)
        self.assertIn("id0001", ids)
        self.assertIn("id0002", ids)

    def test_list_of_dicts(self):
        lst = [{"flow_id": "aaaa11"}, {"flow_id": "bbbb22"}]
        ids = extract_flow_ids(lst)
        self.assertEqual(ids, ["aaaa11", "bbbb22"])

    def test_dedup_and_order_preserved(self):
        ids = extract_flow_ids(
            "flow_id AAAA11",
            {"flow_id": "BBBB22"},
            "flow_id AAAA11 seen again",
            {"flow_id": "CCCC33"},
        )
        self.assertEqual(ids, ["AAAA11", "BBBB22", "CCCC33"])

    def test_no_match_without_keyword(self):
        ids = extract_flow_ids("I saw ABCDEF and QWERTY as tokens.")
        self.assertEqual(ids, [])

    def test_ignores_none_values(self):
        ids = extract_flow_ids(None, "flow_id zz11aa")
        self.assertEqual(ids, ["zz11aa"])

    def test_bare_flow_in_prose_does_not_match(self):
        self.assertEqual(extract_flow_ids("data flow analysis found an issue"), [])
        self.assertEqual(extract_flow_ids("the flow chart shows"), [])
        self.assertEqual(extract_flow_ids("request flow through the system"), [])


if __name__ == "__main__":
    unittest.main()
