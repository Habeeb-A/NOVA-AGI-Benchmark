"""
NOVA Generator — Offline Structure Test
Run this BEFORE using the real API to verify the full pipeline parses and
scores correctly. Uses a hardcoded mock instance — no API key required.

Usage: python nova_test.py
"""

import json
import sys

MOCK_PHYSICS_RESPONSE = {
    "domain_name": "Vornian Field Theory",
    "domain_description": "In the Vornian universe, all matter is composed of vorn particles classified as either plovic or ghentic. These particles interact through invisible trel fields whenever they occupy the same space. The outcome of every interaction is determined entirely by the classification of the two particles involved.",
    "rules": [
        "Rule 1: When two plovic objects meet, they combine into a single plovic object whose mass equals the sum of both.",
        "Rule 2: When a plovic object meets a ghentic object, the ghentic object is destroyed and the plovic object's mass doubles.",
        "Rule 3: When two ghentic objects meet, both are destroyed and a new plovic object of mass 1 is created."
    ],
    "examples": [
        {"input": "A plovic object of mass 3 meets a plovic object of mass 5.", "output": "A single plovic object of mass 8 is created."},
        {"input": "A plovic object of mass 4 meets a ghentic object of mass 7.", "output": "The ghentic object is destroyed; the plovic object now has mass 8."},
        {"input": "A ghentic object of mass 2 meets a ghentic object of mass 9.", "output": "Both are destroyed; a new plovic object of mass 1 is created."},
        {"input": "A plovic object of mass 1 meets a plovic object of mass 1.", "output": "A single plovic object of mass 2 is created."},
        {"input": "A plovic object of mass 10 meets a ghentic object of mass 3.", "output": "The ghentic object is destroyed; the plovic object now has mass 20."},
        {"input": "A ghentic object of mass 5 meets a ghentic object of mass 5.", "output": "Both are destroyed; a new plovic object of mass 1 is created."},
        {"input": "A plovic object of mass 6 meets a plovic object of mass 2.", "output": "A single plovic object of mass 8 is created."},
        {"input": "A plovic object of mass 7 meets a ghentic object of mass 1.", "output": "The ghentic object is destroyed; the plovic object now has mass 14."}
    ],
    "phase_c_probes": [
        {
            "question": "A plovic object of mass 3 meets a plovic object of mass 9. What is the result?",
            "answer_choices": [
                "A single plovic object of mass 12 is created.",
                "A single plovic object of mass 6 is created.",
                "The smaller object is destroyed.",
                "Both objects remain separate."
            ],
            "ground_truth": "A single plovic object of mass 12 is created.",
            "ground_truth_explanation": "Rule 1: plovic + plovic → combined plovic with sum mass (3+9=12)."
        },
        {
            "question": "A plovic object of mass 5 meets a ghentic object of mass 11. What is the result?",
            "answer_choices": [
                "The plovic object is destroyed.",
                "The ghentic object is destroyed; the plovic object now has mass 10.",
                "Both combine into a mass-16 object.",
                "A new ghentic object of mass 6 is created."
            ],
            "ground_truth": "The ghentic object is destroyed; the plovic object now has mass 10.",
            "ground_truth_explanation": "Rule 2: plovic + ghentic → ghentic destroyed, plovic mass doubles (5×2=10)."
        },
        {
            "question": "A ghentic object of mass 4 meets a ghentic object of mass 8. What is the result?",
            "answer_choices": [
                "A single ghentic object of mass 12 is created.",
                "Both are destroyed; a new plovic object of mass 1 is created.",
                "The smaller ghentic is absorbed.",
                "Both remain unchanged."
            ],
            "ground_truth": "Both are destroyed; a new plovic object of mass 1 is created.",
            "ground_truth_explanation": "Rule 3: ghentic + ghentic → both destroyed, new plovic mass 1."
        }
    ],
    "phase_d_probes": [
        {
            "question": "A ghentic object of mass 3 meets a ghentic object of mass 7. The resulting object then meets a plovic object of mass 4. What is the final state?",
            "answer_choices": [
                "A single plovic object of mass 5 is created.",
                "A plovic object of mass 8 exists.",
                "A ghentic object of mass 4 exists.",
                "Two separate objects of mass 1 and 4 exist."
            ],
            "ground_truth": "A single plovic object of mass 5 is created.",
            "ground_truth_explanation": "Step 1 (Rule 3): ghentic+ghentic → plovic(1). Step 2 (Rule 1): plovic(1)+plovic(4) → plovic(5)."
        },
        {
            "question": "A plovic object of mass 2 meets a ghentic object of mass 6. The result then meets another ghentic object of mass 3. What is the final state?",
            "answer_choices": [
                "A single plovic object of mass 8 exists.",
                "A plovic object of mass 4 and a ghentic of mass 3 exist.",
                "Both are destroyed; a new plovic of mass 1 is created.",
                "A single ghentic object of mass 3 exists."
            ],
            "ground_truth": "A single plovic object of mass 8 exists.",
            "ground_truth_explanation": "Step 1 (Rule 2): plovic(2)+ghentic(6) → plovic(4). Step 2 (Rule 2): plovic(4)+ghentic(3) → plovic(8)."
        },
        {
            "question": "Three ghentic objects of mass 2, 5, and 9 interact in pairs. First the mass-2 and mass-5 meet, then the result meets the mass-9. What is the final state?",
            "answer_choices": [
                "A plovic object of mass 2 exists.",
                "A ghentic object of mass 16 exists.",
                "A plovic object of mass 10 exists.",
                "Three separate objects remain."
            ],
            "ground_truth": "A plovic object of mass 2 exists.",
            "ground_truth_explanation": "Step 1 (Rule 3): ghentic(2)+ghentic(5) → plovic(1). Step 2 (Rule 2): plovic(1)+ghentic(9) → plovic(2)."
        }
    ],
    "transfer_domain_name": "Beshian Council System",
    "transfer_domain_description": "In the Beshian political system, council members hold either kelp-rank or wova-rank. When two members interact in a chamber session, the outcome is determined by their respective ranks. The Beshian system has governed inter-chamber relations for generations.",
    "phase_e_probes": [
        {
            "question": "In the Beshian system: kelp+kelp = merge votes; kelp+wova = wova eliminated, kelp votes double; wova+wova = both eliminated, new kelp with 1 vote. A kelp-rank member with 3 votes meets a kelp-rank member with 9 votes. What is the outcome?",
            "answer_choices": [
                "A single kelp-rank member with 12 votes.",
                "A single kelp-rank member with 6 votes.",
                "The lower-vote member is eliminated.",
                "Both members remain independent."
            ],
            "ground_truth": "A single kelp-rank member with 12 votes.",
            "ground_truth_explanation": "Maps to Rule 1 (plovic+plovic): kelp+kelp → merged kelp with 3+9=12 votes."
        },
        {
            "question": "A kelp-rank member with 5 votes meets a wova-rank member with 11 votes. What is the outcome?",
            "answer_choices": [
                "The kelp member is eliminated.",
                "The wova member is eliminated; the kelp member now has 10 votes.",
                "Both members merge into a 16-vote member.",
                "A new wova member with 6 votes is created."
            ],
            "ground_truth": "The wova member is eliminated; the kelp member now has 10 votes.",
            "ground_truth_explanation": "Maps to Rule 2 (plovic+ghentic): kelp+wova → wova eliminated, kelp doubles (5×2=10)."
        },
        {
            "question": "Two wova-rank members (4 votes and 8 votes respectively) meet. What is the outcome?",
            "answer_choices": [
                "A single wova-rank member with 12 votes is created.",
                "Both are eliminated; a new kelp-rank member with 1 vote is created.",
                "The lower-vote wova is absorbed by the larger.",
                "Both members are unaffected."
            ],
            "ground_truth": "Both are eliminated; a new kelp-rank member with 1 vote is created.",
            "ground_truth_explanation": "Maps to Rule 3 (ghentic+ghentic): wova+wova → both eliminated, new kelp(1)."
        }
    ]
}


def test_json_extraction():
    from nova_generator import extract_json
    assert extract_json(json.dumps({"key": "value"}))["key"] == "value"
    assert extract_json('```json\n{"key": "value"}\n```')["key"] == "value"
    assert extract_json('Preamble\n{"key": "value"}\nPostamble')["key"] == "value"
    print("  ✅ extract_json: all formats handled")


def test_validate_instance():
    from nova_generator import validate_instance
    is_valid, note = validate_instance(MOCK_PHYSICS_RESPONSE, "physics")
    assert is_valid or "flagged" in note, f"Unexpected failure: {note}"
    print(f"  ✅ validate_instance: structural checks pass (note: '{note}')")


def test_score_instance():
    from nova_evaluator import score_instance

    instance = {
        "instance_id": "test_001",
        "domain_type": "physics",
        "difficulty": "medium",
        "phase_c_probes": MOCK_PHYSICS_RESPONSE["phase_c_probes"],
        "phase_d_probes": MOCK_PHYSICS_RESPONSE["phase_d_probes"],
        "phase_e_probes": MOCK_PHYSICS_RESPONSE["phase_e_probes"],
    }

    perfect = {
        phase: [{"answer": p["ground_truth"], "confidence": 90}
                for p in MOCK_PHYSICS_RESPONSE[f"{phase}_probes"]]
        for phase in ["phase_c", "phase_d", "phase_e"]
    }
    r = score_instance(instance, perfect)
    assert r["phase_c"]["accuracy"] == 1.0
    assert r["phase_d"]["accuracy"] == 1.0
    assert r["phase_e"]["accuracy"] == 1.0
    assert r["nova_score"] > 0.8
    print(f"  ✅ score_instance (perfect): NOVA={r['nova_score']:.3f}, depth={r['learning_depth']:.3f}")

    all_wrong = {phase: [{"answer": "wrong", "confidence": 50}] * 3
                 for phase in ["phase_c", "phase_d", "phase_e"]}
    r2 = score_instance(instance, all_wrong)
    assert r2["nova_score"] < 0.2
    print(f"  ✅ score_instance (all wrong): NOVA={r2['nova_score']:.3f}")

    degraded = {
        "phase_c": [{"answer": p["ground_truth"], "confidence": 90}
                    for p in MOCK_PHYSICS_RESPONSE["phase_c_probes"]],
        "phase_d": [{"answer": MOCK_PHYSICS_RESPONSE["phase_d_probes"][0]["ground_truth"], "confidence": 60},
                    {"answer": "wrong", "confidence": 40},
                    {"answer": "wrong", "confidence": 40}],
        "phase_e": [{"answer": "wrong", "confidence": 30}] * 3,
    }
    r3 = score_instance(instance, degraded)
    curve = r3["degradation_curve"]
    print(f"  ✅ score_instance (degradation): curve={[f'{x:.2f}' for x in curve]}, depth={r3['learning_depth']:.3f}")


def test_ece():
    from nova_evaluator import compute_ece
    ece_overconf = compute_ece([1.0] * 10, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    assert ece_overconf > 0.4, f"Expected high ECE for overconfident model, got {ece_overconf}"
    print(f"  ✅ compute_ece (overconfident): ECE={ece_overconf:.4f}")
    ece_perfect = compute_ece([0.9] * 9 + [0.1], [1] * 9 + [0])
    print(f"  ✅ compute_ece (well-calibrated): ECE={ece_perfect:.4f}")


def test_mock_structure():
    inst = MOCK_PHYSICS_RESPONSE
    assert len(inst["examples"]) == 8
    assert len(inst["phase_c_probes"]) == 3
    assert len(inst["phase_d_probes"]) == 3
    assert len(inst["phase_e_probes"]) == 3
    for phase in ["phase_c_probes", "phase_d_probes", "phase_e_probes"]:
        for i, probe in enumerate(inst[phase]):
            assert len(probe["answer_choices"]) == 4, f"{phase}[{i}] needs 4 choices"
            assert probe["ground_truth"] in probe["answer_choices"], \
                f"{phase}[{i}] ground_truth not in choices"
    print("  ✅ Mock instance structure: valid")


def main():
    print("\n🧪 NOVA Pipeline Tests (offline — no API calls)\n")
    tests = [
        ("JSON extraction", test_json_extraction),
        ("Instance structure", test_mock_structure),
        ("Score calculation", test_score_instance),
        ("ECE metric", test_ece),
    ]
    passed = 0
    for name, fn in tests:
        print(f"Testing: {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
        print()

    print(f"{'='*40}")
    print(f"Passed: {passed}/{len(tests)} tests")
    if passed == len(tests):
        print("✅ All tests passed. Ready to run:")
        print("  export GEMINI_API_KEY='your_key'")
        print("  python nova_generator.py --count 5 --domain_type physics --output pilot.json")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
