"""Run an OpenLEWS adapter (or any causal LM) over LEWS 1.0 and emit a prediction file.

  python3 harness/run_openlews.py --base Qwen/Qwen2.5-7B --adapter decoverai/OpenLEWS-7B-v1 \
      --out predictions_openlews7b.json

The prompt format and parsing rules here are the canonical ones: dossier + INSTR,
greedy decoding, 8,192-token input window, and the LAST {"probability": ...} JSON
object in the generation. If no valid JSON is produced the prediction is null; the
evaluator excludes it (coverage), never imputes it.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = os.path.join(os.path.dirname(__file__), "..", "tasks", "lews_v1.jsonl")
MAX_INPUT_TOKENS = 8192
PROB_RE = re.compile(r"\{[^{}]*\"probability\"[^{}]*\}")

INSTR = (
    "\n\nForecast whether this will consolidate into a U.S. federal MDL within ~18 months, "
    "using ONLY the information above. Reason step by step, then end with a JSON line "
    '{"probability": <0..1>, "tier": "HIGH|MEDIUM|LOW"}.\n\nANALYSIS:\n'
)


def parse_prob(text: str) -> float | None:
    m = None
    for m in PROB_RE.finditer(text):
        pass
    if m:
        try:
            p = float(json.loads(m.group(0))["probability"])
            return p if 0.0 <= p <= 1.0 else None
        except Exception:
            return None
    return None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--adapter", default=None, help="PEFT adapter repo or path (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--four-bit", action="store_true", help="load base in 4-bit NF4")
    ap.add_argument("--model-name", default=None, help="name recorded in the prediction file")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base)
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    preds = {}
    tasks = [json.loads(l) for l in open(TASKS)]
    for i, t in enumerate(tasks):
        ids = tok(t["dossier"] + INSTR, return_tensors="pt", truncation=True,
                  max_length=MAX_INPUT_TOKENS).to(model.device)
        out = model.generate(**ids, max_new_tokens=700, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        preds[t["id"]] = parse_prob(txt)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(tasks)}", flush=True)

    name = args.model_name or (args.adapter or args.base).rstrip("/").split("/")[-1]
    json.dump({"model": name, "predictions": preds}, open(args.out, "w"), indent=1)
    n_ans = sum(v is not None for v in preds.values())
    print(f"saved {args.out}  (answered {n_ans}/{len(preds)})")


if __name__ == "__main__":
    main()
