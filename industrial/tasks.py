"""Task adapters retain each supplied package's own model and training code."""
from dataclasses import dataclass
import importlib
import math
from pathlib import Path


@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    metric: str
    maximize: bool
    unit: str
    learning_rates: tuple

    def module(self, name):
        return importlib.import_module(f".baseline_{self.name.lower()}.{name}", package=__package__)

    def load(self, checkpoint):
        model = self.module("model").Model.load(checkpoint)
        if model.meta.get("profile") != self.name or model.meta["task"] != self.kind:
            raise ValueError("Checkpoint task does not match requested task")
        if self.name == "B" and model.meta.get("target_unit") != "minutes":
            raise ValueError("B requires the supplied RUL checkpoint in minutes")
        return model

    def digest(self, checkpoint):
        return self.module("model").checkpoint_digest(checkpoint)

    def evaluate(self, checkpoint, validation):
        model = self.load(checkpoint)
        dataset = self.module("data").read_dataset(validation, self.kind, model.meta["classes"], model.meta["paths"])
        return self.module("train").metrics(model, dataset)

    def predict(self, checkpoint, inputs):
        return self.module("predict").predict(self.load(checkpoint), inputs)

    def improved(self, before, after):
        a, b = before[self.metric], after[self.metric]
        if not all(type(v) in (float, int) and math.isfinite(v) for v in (a, b)):
            raise ValueError("Non-finite validation metric")
        return b > a if self.maximize else b < a

    def validate_plan(self, plan):
        if not isinstance(plan, dict) or set(plan) != {"learning_rate", "batch_size", "epochs", "reason"}:
            raise ValueError("Plan must contain only learning_rate, batch_size, epochs, reason")
        if type(plan["learning_rate"]) not in (float, int) or plan["learning_rate"] not in self.learning_rates:
            raise ValueError("Learning rate outside the task's allowed search space")
        if type(plan["batch_size"]) is not int or plan["batch_size"] not in (64, 128, 256):
            raise ValueError("Batch size outside allowed search space")
        if type(plan["epochs"]) is not int or not 1 <= plan["epochs"] <= 5:
            raise ValueError("Epoch budget must be 1..5")
        if not isinstance(plan["reason"], str) or not 1 <= len(plan["reason"]) <= 1500:
            raise ValueError("Plan needs a bounded reason")
        return plan

    def continue_training(self, checkpoint, train, validation, output, plan, stage, on_epoch=None):
        self.validate_plan(plan)
        parent = self.load(checkpoint)
        parent_digest = self.digest(checkpoint)
        report = self.module("train").train_run(
            self.name, train, validation, output, resume=checkpoint, stage_id=stage,
            learning_rate=plan["learning_rate"], batch_size=plan["batch_size"], epochs=plan["epochs"], on_epoch=on_epoch)
        candidate = self.load(Path(output) / "checkpoint.npz")
        if (report["parent_checkpoint_sha256"] != parent_digest
                or report["checkpoint_sha256"] != self.digest(Path(output) / "checkpoint.npz")
                or report["initial_steps"] != parent.meta["steps"]
                or report["initial_epochs"] != parent.meta["epochs"]
                or report["final_steps"] != candidate.meta["steps"]
                or report["final_epochs"] != candidate.meta["epochs"]
                or candidate.meta["learning_rate"] != plan["learning_rate"]
                or candidate.meta["batch_size"] != plan["batch_size"]
                or candidate.meta["steps"] <= parent.meta["steps"]
                or candidate.meta["epochs"] - parent.meta["epochs"] != plan["epochs"]):
            raise ValueError("Continuation provenance check failed")
        for name in ("mean", "scale"):
            if not self.module("model").np.array_equal(getattr(parent, name), getattr(candidate, name)):
                raise ValueError("Continuation changed input normalization")
        for name in ("seed", "target_mean", "target_scale"):
            if parent.meta[name] != candidate.meta[name]:
                raise ValueError("Continuation changed seed or target scaling")
        if self.digest(checkpoint) != parent_digest:
            raise ValueError("Parent checkpoint was modified")
        return report


TASKS = {
    "A": Task("A", "classification", "macro_f1", True, "class", (0.0003, 0.001, 0.003)),
    "B": Task("B", "regression", "rmse", False, "minutes", (0.0003, 0.001, 0.003)),
    "C": Task("C", "regression", "rmse", False, "cycles", (0.0002, 0.0006, 0.002)),
}
