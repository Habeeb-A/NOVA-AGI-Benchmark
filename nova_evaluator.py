NOVA Benchmark Evaluator
========================
Runs any model through NOVA instances and computes the three-component score:
  - Phase C accuracy (near transfer)    — weight 0.20
  - Phase D accuracy (composition)      — weight 0.35
  - Phase E accuracy (structural xfer)  — weight 0.30
  - ECE calibration                     — weight 0.15

Usage:
  python nova_evaluator.py evaluate --dataset nova_dataset.json --model gemini-2.0-flash --output results_flash.json
  python nova_evaluator.py human --dataset nova_dataset.json --human_responses human_baseline.json --output results_human.json
  python nova_evaluator.py compare results_flash.json results_human.json --plot degradation_curves.png
"""

import os
import json
import time
import argparse
import numpy as np
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_TEMPERATURE = 0.2
MAX_RETRIES = 3
PHASE_WEIGHTS = {"phase_c": 0.20, "phase_d": 0.35, "phase_e": 0.30}
ECE_WEIGHT = 0.15
N_ECE_BINS = 10


def init_gemini(api_key: str) -> None:
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key)


@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=2, max=20))
def call_gemini(prompt: str, model_name: str = "gemini-2.0-flash") -> str:
    model = genai.GenerativeModel(model_name)
    config = genai.types.GenerationConfig(
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=512,
    )
    response = model.generate_content(prompt, generation_config=config)
    return response.text.strip()


EVAL_SYSTEM_PROMPT = """You are taking a structured test. You will be given:
1. A description of an invented domain
2. 8 worked examples (no explanations)
3. A multiple-choice question about the domain

Study the examples carefully to infer the rules, then choose EXACTLY ONE answer
and rate your confidence from 0 to 100.

Respond ONLY with this JSON (no preamble):
{
  "answer": "<exact text of your chosen option>",
  "confidence": <integer 0-100>
}"""


def build_eval_prompt(instance: dict, phase: str, probe_index: int) -> str:
    if phase == "phase_e":
        domain_text = (
            f"DOMAIN: {instance['transfer_domain_name']}\n\n"
            f"{instance['transfer_domain_description']}\n\n"
            f"EXAMPLES (these illustrate how this domain works):\n"
        )
        examples = instance.get("transfer_examples", instance["examples"])
    else:
        domain_text = (
            f"DOMAIN: {instance['domain_name']}\n\n"
            f"{instance['domain_description']}\n\n"
            f"EXAMPLES (these illustrate how this domain works):\n"
        )
        examples = instance["examples"]

    example_text = ""
    for j, ex in enumerate(examples):
        example_text += f"  Example {j+1}:\n"
        example_text += f"    Input:  {ex['input']}\n"
        example_text += f"    Output: {ex['output']}\n"

    probes = instance[f"{phase}_probes"]
    probe = probes[probe_index]
    choices_text = "\n".join(
        f"  {chr(65+k)}. {choice}"
        for k, choice in enumerate(probe["answer_choices"])
    )

    return (
        f"{EVAL_SYSTEM_PROMPT}\n\n"
        f"{'='*60}\n"
        f"{domain_text}"
        f"{example_text}\n"
        f"QUESTION:\n{probe['question']}\n\n"
        f"OPTIONS:\n{choices_text}\n\n"
        f"Your answer (JSON only):"
    )


def parse_model_response(raw: str, probe: dict) -> tuple[str, int]:
    import re
    try:
        clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        data = json.loads(clean)
        answer = data.get("answer", "").strip()
        confidence = max(0, min(100, int(data.get("confidence", 50))))
        return answer, confidence
    except Exception:
        pass
    letter_match = re.search(r"\b([A-D])\b", raw)
    if letter_match:
        idx = ord(letter_match.group(1)) - ord("A")
        choices = probe.get("answer_choices", [])
        if 0 <= idx < len(choices):
            return choices[idx], 50
    return "", 50


def compute_ece(confidences: list[float], correctness: list[int], n_bins: int = N_ECE_BINS) -> float:
    if not confidences:
        return 0.0
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correctness[mask].mean() - confidences[mask].mean())
    return float(ece)


def score_instance(instance: dict, responses: dict) -> dict:
    result = {
        "instance_id": instance["instance_id"],
        "domain_type": instance["domain_type"],
        "difficulty": instance.get("difficulty", "unknown"),
        "phase_c": {}, "phase_d": {}, "phase_e": {}
    }
    all_confidences = []
    all_correctness = []

    for phase in ["phase_c", "phase_d", "phase_e"]:
        probes = instance.get(f"{phase}_probes", [])
        phase_responses = responses.get(phase, [])
        correct = []
        confs = []
        for j, probe in enumerate(probes):
            gt = probe["ground_truth"].strip()
            if j < len(phase_responses):
                r = phase_responses[j]
                pred = r.get("answer", "").strip()
                conf = r.get("confidence", 50) / 100.0
            else:
                pred, conf = "", 0.0
            is_correct = int(pred == gt)
            correct.append(is_correct)
            confs.append(conf)
            all_confidences.append(conf)
            all_correctness.append(is_correct)
        result[phase] = {
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
            "correct": correct,
            "confidences": confs,
        }

    c = result["phase_c"]["accuracy"]
    d = result["phase_d"]["accuracy"]
    e = result["phase_e"]["accuracy"]
    ece = compute_ece(all_confidences, all_correctness)

    result["ece"] = ece
    result["nova_score"] = (
        PHASE_WEIGHTS["phase_c"] * c +
        PHASE_WEIGHTS["phase_d"] * d +
        PHASE_WEIGHTS["phase_e"] * e +
        ECE_WEIGHT * (1 - ece)
    )
    result["degradation_curve"] = [c, d, e]
    result["learning_depth"] = e / c if c > 0 else 0.0
    return result


def aggregate_results(instance_results: list[dict]) -> dict:
    if not instance_results:
        return {}

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    by_type = {}
    for dtype in ["physics", "social", "symbolic", "grammar"]:
        subset = [r for r in instance_results if r["domain_type"] == dtype]
        if subset:
            by_type[dtype] = {
                "n": len(subset),
                "nova_score": mean([r["nova_score"] for r in subset]),
                "phase_c": mean([r["phase_c"]["accuracy"] for r in subset]),
                "phase_d": mean([r["phase_d"]["accuracy"] for r in subset]),
                "phase_e": mean([r["phase_e"]["accuracy"] for r in subset]),
                "ece": mean([r["ece"] for r in subset]),
                "learning_depth": mean([r["learning_depth"] for r in subset]),
            }

    all_c = [r["phase_c"]["accuracy"] for r in instance_results]
    all_d = [r["phase_d"]["accuracy"] for r in instance_results]
    all_e = [r["phase_e"]["accuracy"] for r in instance_results]
    all_scores = [r["nova_score"] for r in instance_results]
    all_ece = [r["ece"] for r in instance_results]

    return {
        "n_instances": len(instance_results),
        "nova_score_mean": mean(all_scores),
        "nova_score_std": float(np.std(all_scores)),
        "phase_c_mean": mean(all_c),
        "phase_d_mean": mean(all_d),
        "phase_e_mean": mean(all_e),
        "ece_mean": mean(all_ece),
        "degradation_curve": [mean(all_c), mean(all_d), mean(all_e)],
        "mean_learning_depth": mean([r["learning_depth"] for r in instance_results]),
        "by_domain_type": by_type,
        "instance_results": instance_results,
    }


def evaluate_model(
    dataset: list[dict],
    model_name: str,
    output_path: str,
    delay: float = 1.0,
    max_instances: Optional[int] = None,
) -> dict:
    instances = dataset[:max_instances] if max_instances else dataset
    instance_results = []

    print(f"\n🔬 Evaluating: {model_name} on {len(instances)} NOVA instances")

    for inst in tqdm(instances, desc="Evaluating"):
        responses = {"phase_c": [], "phase_d": [], "phase_e": []}
        for phase in ["phase_c", "phase_d", "phase_e"]:
            probes = inst.get(f"{phase}_probes", [])
            for j, probe in enumerate(probes):
                prompt = build_eval_prompt(inst, phase, j)
                try:
                    raw = call_gemini(prompt, model_name)
                    answer, confidence = parse_model_response(raw, probe)
                except Exception as e:
                    print(f"\n  ⚠ API error on {inst['instance_id']}/{phase}/{j}: {e}")
                    answer, confidence = "", 50
                responses[phase].append({"answer": answer, "confidence": confidence})
                time.sleep(delay)

        result = score_instance(inst, responses)
        instance_results.append(result)
        aggregated = aggregate_results(instance_results)
        aggregated["model"] = model_name
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)

    print_results_summary(aggregated)
    return aggregated


def load_human_responses(human_json_path: str, dataset: list[dict]) -> dict:
    with open(human_json_path) as f:
        raw = json.load(f)

    from collections import defaultdict, Counter
    by_instance = defaultdict(list)
    for entry in raw:
        by_instance[entry["instance_id"]].append(entry)

    inst_map = {inst["instance_id"]: inst for inst in dataset}
    instance_results = []

    for instance_id, participant_responses in by_instance.items():
        if instance_id not in inst_map:
            continue
        inst = inst_map[instance_id]
        avg_responses = {"phase_c": [], "phase_d": [], "phase_e": []}
        for phase in avg_responses:
            for j in range(3):
                answers_j, confs_j = [], []
                for pr in participant_responses:
                    phase_data = pr.get(phase, [])
                    if j < len(phase_data):
                        answers_j.append(phase_data[j].get("answer", ""))
                        confs_j.append(phase_data[j].get("confidence", 50))
                if answers_j:
                    majority = Counter(answers_j).most_common(1)[0][0]
                    avg_conf = sum(confs_j) / len(confs_j)
                    avg_responses[phase].append({"answer": majority, "confidence": int(avg_conf)})

        result = score_instance(inst, avg_responses)
        result["n_participants"] = len(participant_responses)
        instance_results.append(result)

    aggregated = aggregate_results(instance_results)
    aggregated["model"] = "human_baseline"
    return aggregated


def print_results_summary(results: dict) -> None:
    model = results.get("model", "unknown")
    curve = results.get("degradation_curve", [0, 0, 0])
    print(f"\n{'='*55}")
    print(f"  NOVA Results: {model}")
    print(f"{'='*55}")
    print(f"  Instances evaluated : {results.get('n_instances', 0)}")
    print(f"  NOVA Score          : {results.get('nova_score_mean', 0):.3f} ± {results.get('nova_score_std', 0):.3f}")
    print(f"  Phase C (near xfr)  : {curve[0]:.3f}")
    print(f"  Phase D (compose)   : {curve[1]:.3f}")
    print(f"  Phase E (struct xfr): {curve[2]:.3f}")
    print(f"  ECE (calibration)   : {results.get('ece_mean', 0):.3f}  (lower=better)")
    print(f"  Learning Depth      : {results.get('mean_learning_depth', 0):.3f}  (E/C ratio)")
    print()
    for dtype, stats in results.get("by_domain_type", {}).items():
        print(f"    {dtype:10s}  NOVA={stats['nova_score']:.3f}  "
              f"C={stats['phase_c']:.2f}  D={stats['phase_d']:.2f}  E={stats['phase_e']:.2f}")
    print(f"{'='*55}\n")


def compare_results(result_files: list[str]) -> None:
    all_results = []
    for path in result_files:
        with open(path) as f:
            r = json.load(f)
            r["_file"] = path
            all_results.append(r)
    all_results.sort(key=lambda r: r.get("nova_score_mean", 0), reverse=True)

    print(f"\n{'='*75}")
    print(f"  NOVA LEADERBOARD")
    print(f"{'='*75}")
    print(f"  {'Model':<22} {'NOVA':>7} {'C':>7} {'D':>7} {'E':>7} {'ECE':>7} {'Depth':>7}")
    print(f"  {'-'*70}")
    for r in all_results:
        curve = r.get("degradation_curve", [0, 0, 0])
        print(
            f"  {r.get('model', '?'):<22}"
            f"  {r.get('nova_score_mean', 0):>6.3f}"
            f"  {curve[0]:>6.3f}  {curve[1]:>6.3f}  {curve[2]:>6.3f}"
            f"  {r.get('ece_mean', 0):>6.3f}"
            f"  {r.get('mean_learning_depth', 0):>6.3f}"
        )
    print(f"{'='*75}\n")


def plot_degradation_curves(result_files: list[str], output_path: str = "degradation_curves.png") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.style as style
        style.use("seaborn-v0_8-whitegrid")
    except ImportError:
        print("matplotlib not available. Install: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    phases = ["Phase C\n(Near Transfer)", "Phase D\n(Composition)", "Phase E\n(Structural Transfer)"]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#44BBA4", "#3B1F2B"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, path in enumerate(result_files):
        with open(path) as f:
            r = json.load(f)
        curve = r.get("degradation_curve", [0, 0, 0])
        model = r.get("model", f"Model {i+1}")
        is_human = "human" in model.lower()
        ax.plot([0, 1, 2], curve,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=2.5 if is_human else 1.8,
                linestyle="--" if is_human else "-",
                markersize=8, label=model)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(phases, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.25, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax.set_title(
        "NOVA Benchmark: Learning Depth Degradation Curve\n"
        "(Flat curve = genuine learning; steep drop = surface pattern matching)",
        fontsize=12
    )
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"📈 Plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="NOVA Benchmark Evaluator")
    subparsers = parser.add_subparsers(dest="command")

    ep = subparsers.add_parser("evaluate")
    ep.add_argument("--dataset", required=True)
    ep.add_argument("--model", default="gemini-2.0-flash")
    ep.add_argument("--output", default="nova_results.json")
    ep.add_argument("--api_key", default="")
    ep.add_argument("--max_instances", type=int, default=None)
    ep.add_argument("--delay", type=float, default=1.0)

    hp = subparsers.add_parser("human")
    hp.add_argument("--dataset", required=True)
    hp.add_argument("--human_responses", required=True)
    hp.add_argument("--output", default="nova_results_human.json")

    cp = subparsers.add_parser("compare")
    cp.add_argument("results", nargs="+")
    cp.add_argument("--plot", default="degradation_curves.png")

    args = parser.parse_args()

    if args.command == "evaluate":
        init_gemini(args.api_key or API_KEY)
        with open(args.dataset) as f:
            dataset = json.load(f)
        evaluate_model(dataset, args.model, args.output, args.delay, args.max_instances)

    elif args.command == "human":
        with open(args.dataset) as f:
            dataset = json.load(f)
        results = load_human_responses(args.human_responses, dataset)
        print_results_summary(results)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    elif args.command == "compare":
        compare_results(args.results)
        plot_degradation_curves(args.results, args.plot)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
