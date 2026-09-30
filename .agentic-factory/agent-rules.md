# Agent Factory Rules & Constraints for qlora_llm_ft

## Domain-Specific Guardrails
1. **Never Hardcode Weights:** Hyperparameters inside `train.py` or configuration keys inside `merge_adapters.py` must never be adjusted blindly without updating structural verification logs.
2. **GPU Optimization Integrity:** When modifying training logic, ensure `bitsandbytes` configurations (`load_in_4bit=True`) are preserved to prevent Out-Of-Memory (OOM) runtime state crashes.
3. **Dataset Verification:** Any automated feature updating training structures must match the Alpaca schema framework: `{"instruction": "...", "input": "...", "output": "..."}` inside `./dataset/alpaca_data.json`.

## Agent Specific Boundaries
* **IntakeAgent:** Translate issue descriptions into formal functional tests.
* **WorkerAgent:** Isolate changes completely. If updating model evaluation strings in `infer.py`, write a mock verification unit testing the transformation.
* **ValidatorAgent:** Enforce strict python linting (`flake8`). Reject any model adapters or scripts that fail compilation validation hooks.
