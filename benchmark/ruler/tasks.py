"""
RULER task definitions and synthetic data generation.

RULER (Really Urgent Long-context Evaluation for Research) tests 13 task
types across configurable context lengths.  The dataset is generated
synthetically — there is no fixed download corpus.

Reference: https://huggingface.co/datasets/hsiehjackson/RULER
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

RULER_TASKS: List[str] = [
    # Needle-in-a-haystack (single needle)
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    # Needle-in-a-haystack (multi-key)
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    # Needle-in-a-haystack (multi-value)
    "niah_multivalue",
    # Needle-in-a-haystack (multi-query)
    "niah_multiquery",
    # Variable tracking
    "vt",
    # Common words extraction
    "cwe",
    # Frequent words extraction
    "fwe",
    # Question answering
    "qa_1",
    "qa_2",
]

CONTEXT_LENGTHS: List[int] = [4096, 8192, 16384, 32768, 65536, 131072]

_NIAH_TASKS = {t for t in RULER_TASKS if t.startswith("niah_")}
_VT_TASKS = {"vt"}
_CWE_TASKS = {"cwe"}
_FWE_TASKS = {"fwe"}
_QA_TASKS = {"qa_1", "qa_2"}


class TaskType(str, Enum):
    NIAH_SINGLE = "niah_single"
    NIAH_MULTIKEY = "niah_multikey"
    NIAH_MULTIVALUE = "niah_multivalue"
    NIAH_MULTIQUERY = "niah_multiquery"
    VT = "vt"
    CWE = "cwe"
    FWE = "fwe"
    QA = "qa"

    @classmethod
    def from_task_name(cls, name: str) -> "TaskType":
        if name.startswith("niah_single"):
            return cls.NIAH_SINGLE
        if name.startswith("niah_multikey"):
            return cls.NIAH_MULTIKEY
        if name == "niah_multivalue":
            return cls.NIAH_MULTIVALUE
        if name == "niah_multiquery":
            return cls.NIAH_MULTIQUERY
        if name == "vt":
            return cls.VT
        if name == "cwe":
            return cls.CWE
        if name == "fwe":
            return cls.FWE
        if name.startswith("qa_"):
            return cls.QA
        raise ValueError(f"Unknown RULER task name: {name!r}")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TaskInstance:
    """A single RULER task instance ready for evaluation."""

    task_name: str
    context_length: int  # nominal context length bucket (tokens, approximate)
    context_text: str  # the long-context document / filler text
    question: str  # the question to ask after (or about) the context
    answer: Union[str, List[str]]  # ground-truth answer(s)

    # Optional: stable ID derived from content for snapshot keying
    task_id: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        if self.task_id is None:
            # Deterministic ID from task_name + context_length + answer (which
            # is unique per sample because it encodes the hidden needle/value).
            # This ensures different sample_idx values produce different task_ids
            # even when the question text is identical across samples.
            answer_str = (
                "|".join(self.answer) if isinstance(self.answer, list) else self.answer
            )
            digest = hashlib.sha1(
                f"{self.task_name}|{self.context_length}|{answer_str}".encode()
            ).hexdigest()[:12]
            self.task_id = f"{self.task_name}_cl{self.context_length}_{digest}"


# ---------------------------------------------------------------------------
# Synthetic generation helpers
# ---------------------------------------------------------------------------

_LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "A stitch in time saves nine. "
    "All that glitters is not gold. "
    "To be or not to be that is the question. "
    "It was the best of times it was the worst of times. "
)

_WORDS_POOL = [
    "apple", "banana", "cherry", "diamond", "elephant",
    "falcon", "granite", "horizon", "island", "jungle",
    "kettle", "lantern", "mosaic", "nebula", "orchid",
    "palace", "quartz", "radiant", "sapphire", "twilight",
    "umbrella", "velvet", "walnut", "xenon", "yellow", "zephyr",
]


def _filler_text(approx_tokens: int) -> str:
    """Generate filler text of approximately `approx_tokens` tokens.

    Uses whitespace-split proxy: 1 word ≈ 1 token (conservative).
    """
    # Repeat lorem until we have enough words
    words_needed = approx_tokens
    source = _LOREM.split()
    repetitions = (words_needed // len(source)) + 2
    words = (source * repetitions)[:words_needed]
    return " ".join(words)


def _random_needle(seed: int) -> str:
    """Return a deterministic 'needle' string for NIAH tasks."""
    rng = random.Random(seed)
    adj = rng.choice(["secret", "hidden", "magic", "special", "unique"])
    noun = rng.choice(["code", "phrase", "word", "key", "token"])
    value = "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
    return f"The {adj} {noun} is: {value}"


def _extract_needle_value(needle: str) -> str:
    """Extract the value portion from a needle string."""
    return needle.split(": ", 1)[-1].strip()


# ---------------------------------------------------------------------------
# Task generators — all offline, no network required
# ---------------------------------------------------------------------------


def _generate_niah_single(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """Single needle in a haystack."""
    seed = hash((task_name, context_length, sample_idx)) & 0xFFFFFFFF
    needle = _random_needle(seed)
    value = _extract_needle_value(needle)

    # Determine needle depth (where in the context to insert)
    # niah_single_1 = 50%, _2 = 10%, _3 = 90%
    depth_map = {"niah_single_1": 0.5, "niah_single_2": 0.1, "niah_single_3": 0.9}
    depth = depth_map.get(task_name, 0.5)

    filler_tokens = max(0, context_length - 30)
    filler = _filler_text(filler_tokens)
    words = filler.split()
    insert_pos = int(len(words) * depth)
    words.insert(insert_pos, needle)
    context_text = " ".join(words)

    question = "What is the secret code hidden in the text? Provide only the exact value."
    return TaskInstance(
        task_name=task_name,
        context_length=context_length,
        context_text=context_text,
        question=question,
        answer=value,
    )


def _generate_niah_multikey(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """Multiple distinct needles; ask for one specific key."""
    num_keys_map = {"niah_multikey_1": 2, "niah_multikey_2": 4, "niah_multikey_3": 8}
    num_keys = num_keys_map.get(task_name, 2)

    seed_base = hash((task_name, context_length, sample_idx)) & 0xFFFFFFFF
    needles = [_random_needle(seed_base + i) for i in range(num_keys)]
    values = [_extract_needle_value(n) for n in needles]

    filler_tokens = max(0, context_length - 40 * num_keys)
    filler = _filler_text(filler_tokens)
    words = filler.split()

    # Distribute needles evenly
    for i, needle in enumerate(needles):
        pos = int(len(words) * ((i + 1) / (num_keys + 1)))
        words.insert(pos + i, needle)

    context_text = " ".join(words)

    # Ask for the first needle's value (deterministic)
    target_value = values[0]
    target_needle_phrase = needles[0].split(": ")[0].replace("The ", "")
    question = (
        f"What is the value associated with the '{target_needle_phrase}'? "
        "Provide only the exact value."
    )
    return TaskInstance(
        task_name=task_name,
        context_length=context_length,
        context_text=context_text,
        question=question,
        answer=target_value,
    )


def _generate_niah_multivalue(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """One key, multiple values; ask for all values."""
    seed = hash((task_name, context_length, sample_idx)) & 0xFFFFFFFF
    rng = random.Random(seed)
    num_values = 3
    values = [
        "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
        for _ in range(num_values)
    ]
    needles = [f"The magic token number {i + 1} is: {v}" for i, v in enumerate(values)]

    filler_tokens = max(0, context_length - 40 * num_values)
    filler = _filler_text(filler_tokens)
    words = filler.split()

    for i, needle in enumerate(needles):
        pos = int(len(words) * ((i + 1) / (num_values + 1)))
        words.insert(pos + i, needle)

    context_text = " ".join(words)
    question = "List all magic token values mentioned in the text, in order."
    return TaskInstance(
        task_name=task_name,
        context_length=context_length,
        context_text=context_text,
        question=question,
        answer=values,  # list — any one match scores
    )


def _generate_niah_multiquery(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """One needle, multiple question phrasings."""
    seed = hash((task_name, context_length, sample_idx)) & 0xFFFFFFFF
    needle = _random_needle(seed)
    value = _extract_needle_value(needle)

    filler_tokens = max(0, context_length - 30)
    filler = _filler_text(filler_tokens)
    words = filler.split()
    words.insert(len(words) // 2, needle)
    context_text = " ".join(words)

    question = (
        "What special value or code is hidden in the text? "
        "What secret phrase appears in the document? "
        "Provide only the value you find."
    )
    return TaskInstance(
        task_name=task_name,
        context_length=context_length,
        context_text=context_text,
        question=question,
        answer=value,
    )


def _generate_vt(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """Variable tracking — trace a chain of variable assignments."""
    seed = hash((task_name, context_length, sample_idx)) & 0xFFFFFFFF
    rng = random.Random(seed)

    # Build a chain: X0 = val, X1 = X0, X2 = X1, ..., Xn = X(n-1)
    chain_length = max(3, context_length // 2000)
    initial_value = "".join(rng.choices(string.digits, k=4))

    assignments = [f"Let X0 = {initial_value}."]
    for i in range(1, chain_length):
        assignments.append(f"Let X{i} = X{i - 1}.")

    assignment_text = " ".join(assignments)
    filler_tokens = max(10, context_length - len(assignment_text.split()) - 20)
    filler = _filler_text(filler_tokens)

    # Interleave: filler, then assignments at depth 50%
    words = filler.split()
    insert_pos = len(words) // 2
    words.insert(insert_pos, assignment_text)
    context_text = " ".join(words)

    final_var = f"X{chain_length - 1}"
    question = f"What is the value of {final_var}?"
    return TaskInstance(
        task_name=task_name,
        context_length=context_length,
        context_text=context_text,
        question=question,
        answer=initial_value,
    )


def _generate_cwe(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """Common words extraction — find words that appear most frequently."""
    seed = hash((task_name, context_length, sample_idx)) & 0xFFFFFFFF
    rng = random.Random(seed)

    # Pick 3 "common" words that will be sprinkled throughout
    common_words = rng.sample(_WORDS_POOL[:15], 3)

    # Build a word list with controlled frequencies
    num_words = max(50, context_length // 4)
    background = rng.choices(_WORDS_POOL[15:], k=num_words)

    # Insert common words many times
    inserts = common_words * (num_words // 10)
    all_words = background + inserts
    rng.shuffle(all_words)

    context_text = " ".join(all_words)
    question = (
        "What are the three most common words in the text? "
        "List them separated by commas."
    )
    return TaskInstance(
        task_name=task_name,
        context_length=context_length,
        context_text=context_text,
        question=question,
        answer=common_words,
    )


def _generate_fwe(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """Frequent words extraction — similar to CWE but different framing."""
    seed = hash((task_name, context_length, sample_idx)) & 0xFFFFFFFF
    rng = random.Random(seed)

    frequent_words = rng.sample(_WORDS_POOL[:10], 2)
    num_words = max(50, context_length // 4)
    background = rng.choices(_WORDS_POOL[10:], k=num_words)
    inserts = frequent_words * (num_words // 8)
    all_words = background + inserts
    rng.shuffle(all_words)

    context_text = " ".join(all_words)
    question = (
        "Which words appear most frequently in the passage? "
        "List the top 2 most frequent words."
    )
    return TaskInstance(
        task_name=task_name,
        context_length=context_length,
        context_text=context_text,
        question=question,
        answer=frequent_words,
    )


def _generate_qa(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """Minimal synthetic QA task."""
    seed = hash((task_name, context_length, sample_idx)) & 0xFFFFFFFF
    rng = random.Random(seed)

    # Embed a simple fact
    entity = rng.choice(["Alice", "Bob", "Carol", "Dave", "Eve"])
    location = rng.choice(["Paris", "Tokyo", "London", "Sydney", "Cairo"])
    year = rng.randint(1900, 2020)

    fact = f"{entity} was born in {location} in the year {year}."
    filler_tokens = max(10, context_length - 30)
    filler = _filler_text(filler_tokens)
    words = filler.split()

    insert_pos = rng.randint(0, max(0, len(words) - 1))
    words.insert(insert_pos, fact)
    context_text = " ".join(words)

    if task_name == "qa_1":
        question = f"Where was {entity} born?"
        answer = location
    else:  # qa_2
        question = f"In what year was {entity} born?"
        answer = str(year)

    return TaskInstance(
        task_name=task_name,
        context_length=context_length,
        context_text=context_text,
        question=question,
        answer=answer,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_GENERATORS = {
    "niah_single_1": _generate_niah_single,
    "niah_single_2": _generate_niah_single,
    "niah_single_3": _generate_niah_single,
    "niah_multikey_1": _generate_niah_multikey,
    "niah_multikey_2": _generate_niah_multikey,
    "niah_multikey_3": _generate_niah_multikey,
    "niah_multivalue": _generate_niah_multivalue,
    "niah_multiquery": _generate_niah_multiquery,
    "vt": _generate_vt,
    "cwe": _generate_cwe,
    "fwe": _generate_fwe,
    "qa_1": _generate_qa,
    "qa_2": _generate_qa,
}


def generate_synthetic_task(
    task_name: str,
    context_length: int,
    sample_idx: int = 0,
) -> TaskInstance:
    """Generate a synthetic RULER task instance without any external downloads.

    Args:
        task_name: One of the 13 RULER task names in RULER_TASKS.
        context_length: Nominal context length in tokens (approximate).
        sample_idx: Sample index for deterministic variation within a task.

    Returns:
        A fully-populated TaskInstance ready for evaluation.

    Raises:
        ValueError: If task_name is not in RULER_TASKS.
    """
    if task_name not in _GENERATORS:
        raise ValueError(
            f"Unknown task: {task_name!r}. Must be one of: {RULER_TASKS}"
        )
    return _GENERATORS[task_name](task_name, context_length, sample_idx)
