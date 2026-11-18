"""Real SDK framing + synthetic CPU continuation. No provider or platform calls."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "sdk"), str(HERE.parent)]
import contest_submission as submission
from contest_harness import Round, Sample
from contest_harness.errors import HarnessError
from contest_harness.predict import serve_predictions
from industrial.tasks import TASKS
import numpy as np


def fixture(root, name):
    task = TASKS[name]
    config = json.loads((Path(task.module("train").__file__).parent / "configs" / (name + ".json")).read_text())
    inputs = [{"x": i / 10, "y": i % 3} for i in range(24)]
    paths = task.module("data").flatten(inputs[0])[0]
    x = np.array([task.module("data").flatten(row)[1] for row in inputs], dtype=np.float64)
    labels = config.get("classes", [])
    y = np.array([i % len(labels) if labels else i * 2.0 for i in range(24)])
    model = task.module("model").Model(task.kind, paths, labels, x, y, hidden=config["hidden"], seed=42,
                                       learning_rate=task.learning_rates[0], batch_size=128)
    model.meta["profile"] = name
    if name == "B":
        model.meta["target_unit"] = "minutes"
    model.fit(x, y, 2)
    parent_dir = root / "previous"
    parent_dir.mkdir()
    checkpoint = parent_dir / "checkpoint.npz"
    model.save(checkpoint)
    for split in ("train", "validation"):
        rows = [{"id": f"{split}-{i}", "inputs": inputs[i],
                 "answer": labels[int(y[i])] if labels else float(y[i])} for i in range(24)]
        (root / (split + ".jsonl")).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    state = root / "old-state"
    state.mkdir()
    (state / "preserved.txt").write_text("old-state", encoding="utf-8")
    work = root / "workspace"
    work.mkdir()
    sha = task.digest(checkpoint)
    descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": submission.PREDICT_ARGV}
    parent = {"artifact_id": "selected-parent", "artifact": descriptor,
              "files": [{"path": "checkpoint.npz", "sha256": sha, "size_bytes": checkpoint.stat().st_size}]}
    frame = {"type": "start", "protocol": "contest.harness.workspace.v2", "prompt": "OFFICIAL_TEST_PROMPT",
             "prompt_version": "industrial-workspace-flexible-v3", "dataset_key": name, "round_index": 2,
             "task_contract": {"dataset_key": name, "task": task.kind, "metric": task.metric,
                 "metric_direction": "higher" if task.maximize else "lower", "target_unit": task.unit,
                 "public_projection": {"train_path": str(root / "train.jsonl"), "validation_path": str(root / "validation.jsonl")}},
             "model_name": "offline-fixture-only", "max_output_tokens": 1024, "parent_checkpoint_sha256": sha,
             "parent_artifact": parent, "artifacts": [parent], "state_path": str(state), "state_sha256": "old-state-digest",
             "current_validation": {"metrics": {task.metric: 0.5 if task.maximize else 20.0}}}
    return task, frame, inputs, model


class OfflineHost:
    """Controls measurement fixtures; these numbers are not competition scores."""
    def __init__(self, root, frame, task, writer, *, improved=True, truncate_plan=False):
        self.root, self.frame, self.task, self.writer = root, frame, task, writer
        self.improved, self.truncate_plan = improved, truncate_plan
        self.frames, self.model_count, self.candidate, self.record = [], 0, None, None
        self.decision_artifact = "fixture-candidate"
        self.restored_files = {}
        self.empty_measurement = False

    def readline(self):
        message = json.loads(self.writer.getvalue().splitlines()[-1])
        self.frames.append(message)
        kind = message["type"]
        if kind == "round_complete":
            self.record = message["record"]
            return json.dumps({"type": "round_ack", "accepted": True}) + "\n"
        reply = {"id": message["id"]}
        if kind == "history_request":
            reply.update(type="history_response", records=[{"id": "read-history-1", "round_index": 1}])
        elif kind == "model_request":
            self.model_count += 1
            assert self.frame["prompt"] not in json.dumps(message["messages"])
            assert message["max_tokens"] == self.frame["max_output_tokens"]
            value = ({"learning_rate": self.task.learning_rates[0], "batch_size": 128, "epochs": 1, "reason": "synthetic unit fixture"}
                     if self.model_count == 1 else {"selected_artifact_id": self.decision_artifact, "reason": "Use the measured artifact"})
            reply.update(type="model_response", call_id=f"offline-call-{self.model_count}", response={"choices": [{
                "finish_reason": "length" if self.truncate_plan and self.model_count == 1 else "stop",
                "message": {"content": json.dumps(value)}}]})
        elif kind == "candidate_request":
            assert message["parent_artifact_id"] == self.frame["parent_artifact"]["artifact_id"]
            assert message["parent_checkpoint_sha256"] == self.frame["parent_checkpoint_sha256"]
            self.candidate = self.root / "host-measurement-artifact"
            self.candidate.mkdir()
            for item in message["files"]:
                target = self.candidate / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(item["data_base64"], validate=True))
            # Read the genuine trained fixture checkpoint, independently of the
            # controlled host metric used to exercise selection direction.
            self.task.load(self.candidate / "checkpoint.npz")
            before = self.frame["current_validation"]["metrics"][self.task.metric]
            delta = 0.1 if self.task.maximize else -1.0
            after = before + (delta if self.improved else -delta)
            reply.update(type="candidate_response", artifact_id="fixture-candidate", summary={"metrics": {} if self.empty_measurement else {self.task.metric: after}})
        elif kind == "artifact_read_request":
            data = self.restored_files[(message["artifact_id"], message["path"])]
            chunk = data[message["offset"]:message["offset"] + message["length"]]
            reply.update(type="artifact_read_response", data_base64=base64.b64encode(chunk).decode(),
                         eof=message["offset"] + len(chunk) == len(data))
        elif kind == "state_request":
            assert message["parent_state_sha256"] == "old-state-digest"
            reply.update(type="state_response", state_sha256="fixture-state-digest")
        else:
            raise AssertionError("Unexpected protocol frame " + kind)
        return json.dumps(reply) + "\n"


class SubmissionTests(unittest.TestCase):
    def run_round(self, root, name, *, improved=True, truncate_plan=False):
        task, frame, inputs, model = fixture(root, name)
        writer = io.StringIO()
        host = OfflineHost(root, frame, task, writer, improved=improved, truncate_plan=truncate_plan)
        rnd = Round.start(io.StringIO(json.dumps(frame) + "\n"), writer)
        rnd._reader = host
        rnd.parent.root = root / "previous"  # emulates the host's read-only mount.
        with patch.dict(os.environ, {"CONTEST_WORK": str(root / "workspace")}), patch("socket.create_connection", side_effect=AssertionError("No network allowed")):
            submission.train(rnd)
        self.assertTrue(rnd._finished)
        self.assertEqual(host.model_count, 2)
        self.assertEqual(host.record["history_ids"], ["read-history-1"])
        self.assertEqual(host.record["model_call_ids"], ["offline-call-1", "offline-call-2"])
        self.assertEqual((root / "old-state/preserved.txt").read_text(), "old-state")
        self.assertEqual(task.digest(root / "previous/checkpoint.npz"), frame["parent_checkpoint_sha256"])
        candidate = task.load(host.candidate / "checkpoint.npz")
        model.fit(np.array([task.module("data").flatten(row)[1] for row in inputs]),
                  task.module("data").read_dataset(root / "train.jsonl", task.kind, model.meta["classes"])["y"], 1)
        for tensor in model.params:
            for field in ("params", "first", "second"):
                np.testing.assert_array_equal(getattr(model, field)[tensor], getattr(candidate, field)[tensor])
        self.assertEqual(host.record["selected_artifact_id"], "fixture-candidate" if improved else "selected-parent")
        candidate_request = next(row for row in host.frames if row["type"] == "candidate_request")
        self.assertEqual(candidate_request["artifact"]["predict_argv"], ["python3", "/app/harness.py", "predict"])
        self.assertEqual(candidate_request["artifact"]["checkpoint_paths"], ["checkpoint.npz"])
        return task, frame, inputs, host

    def test_real_sdk_round_trains_all_three_tasks_from_actual_parent(self):
        for name in TASKS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                self.run_round(Path(tmp), name)

    def test_host_degradation_retains_parent_despite_model_selecting_candidate(self):
        for name in ("A", "B"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                self.run_round(Path(tmp), name, improved=False)

    def test_wrong_parent_digest_and_wrong_contract_stop_before_model_calls(self):
        for problem in ("digest", "contract"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                task, frame, *_ = fixture(root, "B")
                if problem == "digest":
                    frame["parent_checkpoint_sha256"] = "sha256:" + "0" * 64
                else:
                    frame["task_contract"]["task"] = "classification"
                writer = io.StringIO()
                host = OfflineHost(root, frame, task, writer)
                rnd = Round.start(io.StringIO(json.dumps(frame) + "\n"), writer)
                rnd._reader, rnd.parent.root = host, root / "previous"
                with self.assertRaises((HarnessError, ValueError)):
                    submission.train(rnd)
                self.assertEqual(host.model_count, 0)
                self.assertIsNone(host.candidate)

    def test_truncated_model_response_never_trains_or_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, frame, *_ = fixture(root, "A")
            writer = io.StringIO()
            host = OfflineHost(root, frame, task, writer, truncate_plan=True)
            rnd = Round.start(io.StringIO(json.dumps(frame) + "\n"), writer)
            rnd._reader, rnd.parent.root = host, root / "previous"
            with self.assertRaisesRegex(ValueError, "MODEL_FINAL_JSON_REQUIRED"):
                submission.train(rnd)
            self.assertEqual(host.model_count, 1)
            self.assertIsNone(host.candidate)

    def test_post_selection_performance_alert_can_restore_and_select_real_older_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, frame, inputs, model = fixture(root, "C")
            old_bytes = (root / "previous/checkpoint.npz").read_bytes()
            # Synthetic degraded deployed model. Earlier artifact is genuinely
            # restored, loaded and evaluated against the current public rows.
            model.params["b2"] += 1000
            current = root / "previous/degraded.npz"
            model.save(current)
            (root / "previous/checkpoint.npz").write_bytes(current.read_bytes())
            frame["parent_checkpoint_sha256"] = task.digest(root / "previous/checkpoint.npz")
            frame["parent_artifact"]["files"][0]["sha256"] = frame["parent_checkpoint_sha256"]
            descriptor = {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": submission.PREDICT_ARGV}
            files = [{"path": "checkpoint.npz", "size_bytes": len(old_bytes), "sha256": "sha256:" + hashlib.sha256(old_bytes).hexdigest()}]
            proof = json.dumps({"files": files, "artifact": descriptor}, sort_keys=True, separators=(",", ":")).encode()
            frame["artifacts"].append({"artifact_id": "older-live-model", "files": files, "artifact": descriptor,
                                        "artifact_sha256": "sha256:" + hashlib.sha256(proof).hexdigest()})
            (root / "old-state/zhixun-last-round.json").write_text(json.dumps({"dataset_key": "C",
                "selected_artifact_id": "selected-parent", "fallback_artifact_id": "older-live-model", "selected_host_validation": 1.0}), encoding="utf-8")
            writer = io.StringIO()
            host = OfflineHost(root, frame, task, writer)
            host.decision_artifact = "older-live-model"
            host.restored_files[("older-live-model", "checkpoint.npz")] = old_bytes
            rnd = Round.start(io.StringIO(json.dumps(frame) + "\n"), writer)
            rnd._reader, rnd.parent.root = host, root / "previous"
            with patch.dict(os.environ, {"CONTEST_WORK": str(root / "workspace")}):
                submission.train(rnd)
            self.assertEqual(host.record["selected_artifact_id"], "older-live-model")
            self.assertTrue(any(message["type"] == "artifact_read_request" for message in host.frames))
            state_request = next(message for message in host.frames if message["type"] == "state_request")
            saved = next(item for item in state_request["files"] if item["path"] == "zhixun-last-round.json")
            record = json.loads(base64.b64decode(saved["data_base64"]))
            self.assertTrue(record["rollback_option"]["improves_current_parent_on_current_public_validation"])
            self.assertEqual(record["rollback_option"]["checkpoint_sha256"], files[0]["sha256"])

    def test_absent_optional_host_metrics_are_not_invented_as_platform_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, frame, *_ = fixture(root, "B")
            writer = io.StringIO()
            host = OfflineHost(root, frame, task, writer)
            host.empty_measurement = True
            # The SDK's start schema does not require current_validation.
            reader_frame = dict(frame)
            reader_frame.pop("current_validation")
            rnd = Round.start(io.StringIO(json.dumps(reader_frame) + "\n"), writer)
            rnd._reader, rnd.parent.root = host, root / "previous"
            with patch.dict(os.environ, {"CONTEST_WORK": str(root / "workspace")}):
                submission.train(rnd)
            self.assertEqual(host.record["configuration"]["selection_comparison_source"], "local_public_validation")
            state_request = next(message for message in host.frames if message["type"] == "state_request")
            saved = next(item for item in state_request["files"] if item["path"] == "zhixun-last-round.json")
            state = json.loads(base64.b64decode(saved["data_base64"]))
            self.assertIsNone(state["host_before"])
            self.assertIsNone(state["host_after"])

    def test_prediction_replies_to_each_row_without_waiting_for_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, frame, inputs, host = self.run_round(root, "C")
            context = {"protocol": "contest.predict-context.v1", "dataset_key": "C", "task": "regression",
                       "artifact_id": "fixture-candidate", "artifact": {"checkpoint_paths": ["checkpoint.npz"]}}
            output = io.StringIO()
            def requests():
                yield json.dumps({"task": "regression", "inputs": inputs[0]}) + "\n"
                self.assertEqual(len(output.getvalue().splitlines()), 1)
                yield json.dumps({"task": "regression", "inputs": inputs[1]}) + "\n"
            with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(host.candidate / "checkpoint.npz"),
                                         "CONTEST_PREDICTION_CONTEXT": json.dumps(context),
                                         "CONTEST_DATASET_KEY": "C", "CONTEST_TASK": "regression"}), \
                    patch("contest_harness.predict.model_dir", return_value=host.candidate):
                serve_predictions(submission.predict, requests(), output)
            results = [json.loads(row) for row in output.getvalue().splitlines()]
            self.assertEqual(results, [task.predict(host.candidate / "checkpoint.npz", row) for row in inputs[:2]])

    def test_fixed_template_describe_imports_real_submission(self):
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONPATH": os.pathsep.join((str(HERE), str(HERE / "sdk"), str(HERE.parent)))}
        result = subprocess.run([sys.executable, str(HERE / "harness.py"), "describe"], env=env,
                                capture_output=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"protocol": "contest.harness.workspace.v2", "model_access": "host_gateway_rpc",
                         "sdk": "contest_harness", "sdk_version": "1.0.2", "tasks": ["classification", "regression"]})


if __name__ == "__main__":
    unittest.main()
