"""
NOVA Benchmark Generator
========================
Novel domain Onboarding and Verification of Adaptation

Generates procedural micro-domain instances for the Kaggle Measuring AGI:
Learning track submission. Uses Gemini API to produce contamination-proof
novel domains that cannot exist in any model's training data.

Each instance contains:
  - domain_description  : rules of an invented domain
  - examples            : 8 worked examples (no explanations)
  - phase_c_probes      : 3 near-transfer probes + ground_truth
  - phase_d_probes      : 3 composition probes + ground_truth
  - phase_e_probes      : 3 structural-transfer probes + ground_truth
  - transfer_domain     : isomorphic domain in different surface vocabulary
  - metadata            : domain_type, rule_count, difficulty, etc.

Usage:
  python nova_generator.py --count 25 --domain_type all --output nova_pilot.json
  python nova_generator.py --count 10 --domain_type physics --output nova_physics.json

Requirements:
  pip install google-generativeai tenacity tqdm
  export GEMINI_API_KEY="your_key_here"

Author: Generated for NOVA benchmark (Kaggle Measuring AGI submission)
"""

import os
import json
import random
import argparse
import time
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.0-flash"
GENERATION_TEMPERATURE = 0.9   # High entropy for novelty
VALIDATION_TEMPERATURE = 0.1   # Low entropy for ground-truth checking

DOMAIN_TYPES = ["physics", "social", "symbolic", "grammar"]

# Invented vocabulary banks — prevents real-world term leakage
INVENTED_NOUNS = [
    "vorn", "kelp", "mira", "zara", "tolu", "besh", "wova", "pril",
    "chent", "dreva", "folu", "gwim", "havet", "ilmo", "juxo", "kref",
    "lurba", "mysti", "norva", "olek", "pluve", "quon", "resti", "suval",
    "twemo", "uxor", "vladi", "wexpo", "xalni", "yoval", "zephi", "arko",
    "brovi", "crelu", "doxal", "envri", "faxil", "glopu", "hovic", "ixamo"
]

INVENTED_VERBS = [
    "trels", "vomp", "zinks", "blavi", "churt", "dreps", "floop", "grank",
    "helst", "invok", "javel", "klurt", "moven", "nexal", "oprev", "plext",
    "quonk", "rivel", "stelp", "tuxan", "uvres", "vompt", "wexal", "xyven"
]

INVENTED_ADJECTIVES = [
    "vrellic", "omban", "stuxal", "plovic", "ghentic", "dravish",
    "kurven", "molpan", "nextic", "quelvi", "ristov", "selpan",
    "torvik", "uvran", "velstric", "wompan", "xalvic", "yestron"
]

CULTURAL_CONTEXTS = [
    "Oloba", "Vrenshi", "Kaltamu", "Zephira", "Brovan", "Luxteri",
    "Morvak", "Neshpal", "Quelov", "Restami", "Sukthi", "Twemora"
]


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Probe:
    question: str
    answer_choices: list[str]
    ground_truth: str
    ground_truth_explanation: str


@dataclass
class NovaInstance:
    instance_id: str
    domain_type: str
    domain_name: str
    domain_description: str
    examples: list[dict]
    phase_c_probes: list[dict]
    phase_d_probes: list[dict]
    phase_e_probes: list[dict]
    transfer_domain_name: str
    transfer_domain_description: str
    rules: list[str]
    difficulty: str
    generation_model: str
    validated: bool = False
    validation_notes: str = ""


# ─────────────────────────────────────────────
# Gemini API wrapper
# ─────────────────────────────────────────────

def init_gemini(api_key: str) -> None:
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Export it: export GEMINI_API_KEY='your_key'"
        )
    genai.configure(api_key=api_key)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def call_gemini(prompt: str, temperature: float = 0.9, max_tokens: int = 4096) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        candidate_count=1,
    )
    response = model.generate_content(prompt, generation_config=config)
    return response.text.strip()


def extract_json(text: str) -> dict | list:
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Could not extract JSON from model output:\n{text[:500]}...")


# ─────────────────────────────────────────────
# Domain-specific prompt builders
# ─────────────────────────────────────────────

def _sample_vocabulary(n: int, bank: list) -> list[str]:
    return random.sample(bank, min(n, len(bank)))


def build_physics_prompt(seed_vocab: dict) -> str:
    nouns = seed_vocab["nouns"]
    verbs = seed_vocab["verbs"]
    adjs = seed_vocab["adjs"]

    return f"""You are designing a NOVEL FICTIONAL PHYSICS SYSTEM for an AI benchmark.

CRITICAL REQUIREMENTS:
1. The domain must be entirely invented — no real physics (gravity, magnetism, thermodynamics, etc.)
2. Use ONLY these invented words for objects and interactions:
   Objects: {nouns}
   Interactions: {verbs}
   Properties: {adjs}
3. Define exactly 3 rules. Each rule must have a clear precondition and outcome.
4. Rules must interact so that combining two rules produces non-obvious results (needed for Phase D).
5. The rules must be learnable from 8 examples without verbal explanation.

OUTPUT exactly this JSON structure (no preamble, no trailing text):

{{
  "domain_name": "<invented name using vocab above>",
  "domain_description": "<2-3 sentences describing what this domain is about — no rules stated, just context>",
  "rules": [
    "<Rule 1: complete statement with precondition and outcome>",
    "<Rule 2: complete statement with precondition and outcome>",
    "<Rule 3: complete statement with precondition and outcome>"
  ],
  "examples": [
    {{"input": "<scenario using invented vocab>", "output": "<correct result under rules>"}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}}
  ],
  "phase_c_probes": [
    {{
      "question": "<new scenario, same rules, different objects — tests single rule application>",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact string matching the correct choice>",
      "ground_truth_explanation": "<which rule this tests and why the answer is correct>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ],
  "phase_d_probes": [
    {{
      "question": "<scenario requiring TWO rules applied in sequence — the second application depends on result of first>",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact string>",
      "ground_truth_explanation": "<which rules combine and how>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ],
  "transfer_domain_name": "<different invented name>",
  "transfer_domain_description": "<2-3 sentences: COMPLETELY different surface vocabulary and context, but same abstract rule structure as the original domain>",
  "phase_e_probes": [
    {{
      "question": "<structurally identical to a Phase C probe but in transfer domain surface>",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact string>",
      "ground_truth_explanation": "<maps to which original rule, proving structural transfer>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ]
}}"""


def build_social_prompt(seed_vocab: dict) -> str:
    culture = seed_vocab["culture"]
    nouns = seed_vocab["nouns"]
    verbs = seed_vocab["verbs"]
    transfer_culture = seed_vocab["transfer_culture"]

    return f"""You are designing a NOVEL FICTIONAL CULTURAL RULE SYSTEM for an AI benchmark.

CRITICAL REQUIREMENTS:
1. Invent an entirely fictional society called "{culture}" with 3 social obligation rules.
2. Rules must involve conditions like time-of-day, relationships, weather, or social context.
3. Rules must NOT mirror any real cultural norm that an AI might recognize.
4. Use these invented terms for objects/roles: {nouns}
5. The rules must create non-obvious outcomes when combined (for Phase D).
6. Transfer domain: create "{transfer_culture}" with the SAME ABSTRACT RULE STRUCTURE but completely different surface (different obligations, triggers, outcomes — same logical skeleton).

OUTPUT exactly this JSON structure:

{{
  "domain_name": "{culture}",
  "domain_description": "<2-3 sentences: what kind of society this is — no rules stated>",
  "rules": [
    "<Rule 1: condition → social obligation/outcome>",
    "<Rule 2: condition → social obligation/outcome>",
    "<Rule 3: condition → social obligation/outcome>"
  ],
  "examples": [
    {{"input": "<social scenario with named characters and conditions>", "output": "<correct social outcome under rules>"}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}}
  ],
  "phase_c_probes": [
    {{
      "question": "<new scenario, tests one rule clearly>",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact string>",
      "ground_truth_explanation": "<which rule and why>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ],
  "phase_d_probes": [
    {{
      "question": "<scenario where two rules create conflicting or compounding obligations>",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact string>",
      "ground_truth_explanation": "<rule interaction explanation>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ],
  "transfer_domain_name": "{transfer_culture}",
  "transfer_domain_description": "<completely different surface obligations and contexts, same 3-rule abstract structure as {culture}>",
  "phase_e_probes": [
    {{
      "question": "<structurally parallel to Phase C probe 1 but in {transfer_culture} surface>",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact string>",
      "ground_truth_explanation": "<maps to which {culture} rule>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ]
}}"""


def build_symbolic_prompt(seed_vocab: dict) -> str:
    symbols = seed_vocab["symbols"]
    nouns = seed_vocab["nouns"]

    return f"""You are designing a NOVEL SYMBOLIC OPERATION SYSTEM for an AI benchmark.

CRITICAL REQUIREMENTS:
1. Invent 3 fictional binary operators using these symbols: {symbols}
2. Each operator takes two numeric values and produces a result via an invented rule.
3. The rules must NOT be standard arithmetic (no +, -, *, /, mod, etc. in disguise).
4. Rules can involve invented conditions: e.g., "if left > right, result is X; else Y".
5. Operator interaction must be non-trivial: (a ⊕ b) ◈ c ≠ a ⊕ (b ◈ c) ideally.
6. Present rules ONLY through examples — never state the rule explicitly in the description.
7. Transfer domain: same abstract operator logic, different symbols and number ranges.

Use numeric values in range 1–20 for all examples and probes.

OUTPUT exactly this JSON structure:

{{
  "domain_name": "<invented system name using nouns: {nouns}>",
  "domain_description": "<2-3 sentences: what kind of symbolic system this is — no operator rules stated>",
  "rules": [
    "<Operator 1 {symbols[0]}: full definition>",
    "<Operator 2 {symbols[1]}: full definition>",
    "<Operator 3 {symbols[2]}: full definition>"
  ],
  "examples": [
    {{"input": "What is 7 {symbols[0]} 3?", "output": "<correct numeric result>"}},
    {{"input": "What is 4 {symbols[1]} 9?", "output": "<correct numeric result>"}},
    {{"input": "What is 12 {symbols[2]} 5?", "output": "<correct numeric result>"}},
    {{"input": "What is 2 {symbols[0]} 2?", "output": "<correct numeric result>"}},
    {{"input": "What is 15 {symbols[1]} 6?", "output": "<correct numeric result>"}},
    {{"input": "What is 8 {symbols[2]} 8?", "output": "<correct numeric result>"}},
    {{"input": "What is 1 {symbols[0]} 10?", "output": "<correct numeric result>"}},
    {{"input": "What is 11 {symbols[1]} 4?", "output": "<correct numeric result>"}}
  ],
  "phase_c_probes": [
    {{
      "question": "What is 5 {symbols[0]} 13?",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact numeric string>",
      "ground_truth_explanation": "<operator 1 rule applied step by step>"
    }},
    {{
      "question": "What is 9 {symbols[2]} 3?",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact numeric string>",
      "ground_truth_explanation": "<operator 3 rule applied>"
    }},
    {{
      "question": "What is 6 {symbols[1]} 6?",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact numeric string>",
      "ground_truth_explanation": "<operator 2 rule applied — especially interesting for equal-value case>"
    }}
  ],
  "phase_d_probes": [
    {{
      "question": "What is (3 {symbols[0]} 7) {symbols[1]} 4?",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact numeric string>",
      "ground_truth_explanation": "<step 1: apply op1, get X; step 2: apply op2 to X and 4>"
    }},
    {{
      "question": "What is 2 {symbols[2]} (8 {symbols[0]} 5)?",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact numeric string>",
      "ground_truth_explanation": "<step 1: apply op1 to 8,5; step 2: apply op3 to 2 and result>"
    }},
    {{
      "question": "What is (10 {symbols[1]} 3) {symbols[2]} (1 {symbols[0]} 9)?",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact numeric string>",
      "ground_truth_explanation": "<three-operator composition>"
    }}
  ],
  "transfer_domain_name": "<different system name>",
  "transfer_domain_description": "<same abstract operator structure, different symbols ★ ◆ ▲ and number range 1-30>",
  "phase_e_probes": [
    {{
      "question": "<structurally parallel to Phase C probe 1 but using ★ instead of {symbols[0]} with different numbers>",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact numeric string>",
      "ground_truth_explanation": "<same rule structure as op1, proved by mapping>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ]
}}"""


def build_grammar_prompt(seed_vocab: dict) -> str:
    lang_name = seed_vocab["lang_name"]
    words = seed_vocab["words"]
    transfer_lang = seed_vocab["transfer_lang"]

    return f"""You are designing a MINIATURE INVENTED LANGUAGE GRAMMAR for an AI benchmark.

CRITICAL REQUIREMENTS:
1. Create a language called "{lang_name}" with exactly 3 grammatical rules.
2. Choose rules from: word order (SOV/VSO/OVS etc.), case marking, verb agreement, tense markers, evidentiality markers, or topic prominence.
3. Use ONLY these invented words: {words}
4. Provide 8 example sentence pairs (invented language → English gloss).
5. Rules must interact: a sentence testing rule 3 may also require applying rule 1 first.
6. Transfer domain: language "{transfer_lang}" with same grammatical rules but entirely different vocabulary.

OUTPUT exactly this JSON structure:

{{
  "domain_name": "{lang_name}",
  "domain_description": "<2-3 sentences: what kind of language this appears to be — no grammatical rules stated>",
  "vocabulary": {{
    "nouns": {{"<word1>": "<English gloss>", "<word2>": "<English gloss>", "<word3>": "<English gloss>", "<word4>": "<English gloss>"}},
    "verbs": {{"<word5>": "<English gloss>", "<word6>": "<English gloss>", "<word7>": "<English gloss>"}},
    "markers": {{"<word8>": "<function>", "<word9>": "<function>"}}
  }},
  "rules": [
    "<Grammatical rule 1>",
    "<Grammatical rule 2>",
    "<Grammatical rule 3>"
  ],
  "examples": [
    {{"input": "<{lang_name} sentence>", "output": "<English translation>"}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}},
    {{"input": "...", "output": "..."}}
  ],
  "phase_c_probes": [
    {{
      "question": "Translate this {lang_name} sentence into English: '<new sentence>'",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact correct English>",
      "ground_truth_explanation": "<which rule>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ],
  "phase_d_probes": [
    {{
      "question": "Translate into {lang_name}: '<English sentence requiring two rules>'",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact correct {lang_name}>",
      "ground_truth_explanation": "<both rules required>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ],
  "transfer_domain_name": "{transfer_lang}",
  "transfer_domain_description": "<same 3 rules, completely different vocabulary>",
  "transfer_vocabulary": {{
    "nouns": {{"<word>": "<gloss>"}},
    "verbs": {{"<word>": "<gloss>"}},
    "markers": {{"<word>": "<function>"}}
  }},
  "phase_e_probes": [
    {{
      "question": "Translate this {transfer_lang} sentence into English: '<sentence>'",
      "answer_choices": ["<correct>", "<wrong1>", "<wrong2>", "<wrong3>"],
      "ground_truth": "<exact correct English>",
      "ground_truth_explanation": "<maps to rule N>"
    }},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}},
    {{"question": "...", "answer_choices": ["...", "...", "...", "..."], "ground_truth": "...", "ground_truth_explanation": "..."}}
  ]
}}"""


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_instance(instance_data: dict, domain_type: str) -> tuple[bool, str]:
    required_keys = [
        "domain_name", "domain_description", "rules", "examples",
        "phase_c_probes", "phase_d_probes",
        "transfer_domain_name", "transfer_domain_description",
        "phase_e_probes"
    ]
    missing = [k for k in required_keys if k not in instance_data]
    if missing:
        return False, f"Missing keys: {missing}"

    if len(instance_data["examples"]) < 8:
        return False, f"Only {len(instance_data['examples'])} examples (need 8)"

    for phase in ["phase_c_probes", "phase_d_probes", "phase_e_probes"]:
        probes = instance_data.get(phase, [])
        if len(probes) < 3:
            return False, f"{phase} has only {len(probes)} probes (need 3)"
        for i, probe in enumerate(probes):
            if "answer_choices" not in probe or len(probe["answer_choices"]) != 4:
                return False, f"{phase}[{i}] doesn't have exactly 4 answer choices"
            if probe.get("ground_truth") not in probe.get("answer_choices", []):
                return False, f"{phase}[{i}] ground_truth not in answer_choices"

    rules = instance_data.get("rules", [])
    examples = instance_data.get("examples", [])
    validation_prompt = f"""You are checking an AI benchmark item for internal consistency.

RULES:
{json.dumps(rules, indent=2)}

FIRST 3 EXAMPLES:
{json.dumps(examples[:3], indent=2)}

FIRST PHASE D PROBE:
{json.dumps(instance_data['phase_d_probes'][0], indent=2)}

TASK: Verify the ground_truth of the Phase D probe is correct given the rules.
Answer ONLY with this JSON: {{"consistent": true/false, "note": "<one sentence>"}}"""

    try:
        response = call_gemini(validation_prompt, temperature=VALIDATION_TEMPERATURE, max_tokens=200)
        result = extract_json(response)
        return result.get("consistent", False), result.get("note", "")
    except Exception as e:
        return True, f"Validation call failed ({e}), flagged for human review"


# ─────────────────────────────────────────────
# Domain seeds
# ─────────────────────────────────────────────

def generate_seed_vocab(domain_type: str) -> dict:
    if domain_type == "physics":
        return {
            "nouns": _sample_vocabulary(5, INVENTED_NOUNS),
            "verbs": _sample_vocabulary(4, INVENTED_VERBS),
            "adjs": _sample_vocabulary(4, INVENTED_ADJECTIVES),
        }
    elif domain_type == "social":
        cultures = random.sample(CULTURAL_CONTEXTS, 2)
        return {
            "culture": cultures[0],
            "transfer_culture": cultures[1],
            "nouns": _sample_vocabulary(6, INVENTED_NOUNS),
            "verbs": _sample_vocabulary(3, INVENTED_VERBS),
        }
    elif domain_type == "symbolic":
        symbol_pool = ["⊕", "◈", "⊗", "⊘", "⊛", "⊜", "⊝", "⋄", "⋆", "⋇"]
        return {
            "symbols": random.sample(symbol_pool, 3),
            "nouns": _sample_vocabulary(3, INVENTED_NOUNS),
        }
    elif domain_type == "grammar":
        lang_names = [
            f"{random.choice(INVENTED_NOUNS).capitalize()}an",
            f"{random.choice(INVENTED_NOUNS).capitalize()}ic",
        ]
        words = _sample_vocabulary(12, INVENTED_NOUNS + INVENTED_VERBS)
        return {
            "lang_name": lang_names[0],
            "transfer_lang": lang_names[1],
            "words": words,
        }
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")


# ─────────────────────────────────────────────
# Main generation
# ─────────────────────────────────────────────

def generate_instance(domain_type: str, instance_id: str) -> Optional[NovaInstance]:
    seed = generate_seed_vocab(domain_type)
    prompt_builders = {
        "physics": build_physics_prompt,
        "social": build_social_prompt,
        "symbolic": build_symbolic_prompt,
        "grammar": build_grammar_prompt,
    }
    prompt = prompt_builders[domain_type](seed)

    try:
        raw_response = call_gemini(prompt, temperature=GENERATION_TEMPERATURE)
        data = extract_json(raw_response)
    except Exception as e:
        print(f"\n  ⚠ Generation failed for {instance_id}: {e}")
        return None

    is_valid, note = validate_instance(data, domain_type)

    instance = NovaInstance(
        instance_id=instance_id,
        domain_type=domain_type,
        domain_name=data.get("domain_name", f"Unknown_{instance_id}"),
        domain_description=data.get("domain_description", ""),
        examples=data.get("examples", []),
        phase_c_probes=data.get("phase_c_probes", []),
        phase_d_probes=data.get("phase_d_probes", []),
        phase_e_probes=data.get("phase_e_probes", []),
        transfer_domain_name=data.get("transfer_domain_name", ""),
        transfer_domain_description=data.get("transfer_domain_description", ""),
        rules=data.get("rules", []),
        difficulty=_infer_difficulty(data),
        generation_model=MODEL_NAME,
        validated=is_valid,
        validation_notes=note,
    )
    return instance


def _infer_difficulty(data: dict) -> str:
    rules = data.get("rules", [])
    d_probes = data.get("phase_d_probes", [])
    max_operators = max(
        (probe.get("ground_truth_explanation", "").lower().count("rule") +
         probe.get("ground_truth_explanation", "").lower().count("step"))
        for probe in d_probes
    ) if d_probes else 0
    if len(rules) >= 3 and max_operators >= 4:
        return "hard"
    elif len(rules) >= 2 and max_operators >= 2:
        return "medium"
    return "easy"


def generate_dataset(
    count: int,
    domain_type: str = "all",
    output_path: str = "nova_dataset.json",
    delay_between_calls: float = 1.5,
) -> list[dict]:
    if domain_type == "all":
        types = DOMAIN_TYPES * (count // len(DOMAIN_TYPES) + 1)
        types = types[:count]
        random.shuffle(types)
    else:
        types = [domain_type] * count

    instances = []
    failed = 0

    print(f"\n🚀 Generating {count} NOVA instances (model: {MODEL_NAME})")
    print(f"   Domain type(s): {domain_type}")
    print(f"   Output: {output_path}\n")

    for i, dtype in enumerate(tqdm(types, desc="Generating")):
        instance_id = f"nova_{dtype}_{i+1:04d}"
        instance = generate_instance(dtype, instance_id)
        if instance is None:
            failed += 1
            continue
        instance_dict = asdict(instance)
        instances.append(instance_dict)
        with open(output_path, "w") as f:
            json.dump(instances, f, indent=2, ensure_ascii=False)
        time.sleep(delay_between_calls)

    print(f"\n✅ Complete: {len(instances)} instances generated, {failed} failed")
    print(f"   Saved to: {output_path}")
    return instances


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NOVA Benchmark Generator — Kaggle Measuring AGI (Learning Track)"
    )
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--domain_type", type=str, default="all",
                        choices=["all", "physics", "social", "symbolic", "grammar"])
    parser.add_argument("--output", type=str, default="nova_dataset.json")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--validate_only", type=str, default="")

    args = parser.parse_args()
    api_key = args.api_key or API_KEY
    init_gemini(api_key)

    if args.validate_only:
        with open(args.validate_only) as f:
            existing = json.load(f)
        for inst in tqdm(existing, desc="Validating"):
            is_valid, note = validate_instance(inst, inst["domain_type"])
            inst["validated"] = is_valid
            inst["validation_notes"] = note
        with open(args.validate_only, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Done. Re-saved to {args.validate_only}")
    else:
        generate_dataset(
            count=args.count,
            domain_type=args.domain_type,
            output_path=args.output,
            delay_between_calls=args.delay,
        )


if __name__ == "__main__":
    main()
