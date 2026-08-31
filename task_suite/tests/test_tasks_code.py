"""Cross-check the authored coding problems against reference solutions.

Every problem's expected values were written by hand next to its prompt.
The reference solutions below were written separately, from the prompt text
rather than from the expectations. If the two agree on every case, both are
almost certainly right; a disagreement means one of them is wrong and the
suite is not safe to train against yet. This is the check that keeps a typo
in a test case from silently capping a whole category's reward.

The solutions run through a plain subprocess, not a sandbox: this is our own
fixture code, not model output. Module 3 owns the untrusted path.
"""

import subprocess
import sys

import pytest

from task_suite.tasks_code import PROBLEMS, generate_code_tasks
from task_suite.verifiers import verify
from task_suite.verify_code import CodeRunResult

REFERENCE: dict[str, str] = {
    "sum_of_evens": """
def sum_of_evens(numbers):
    return sum(n for n in numbers if n % 2 == 0)
""",
    "count_vowels": """
def count_vowels(text):
    return sum(1 for c in text.lower() if c in "aeiou")
""",
    "reverse_words": """
def reverse_words(sentence):
    return " ".join(reversed(sentence.split()))
""",
    "is_palindrome": """
def is_palindrome(text):
    kept = [c.lower() for c in text if c.isalnum()]
    return kept == kept[::-1]
""",
    "second_largest": """
def second_largest(numbers):
    unique = sorted(set(numbers), reverse=True)
    return unique[1] if len(unique) > 1 else None
""",
    "running_total": """
def running_total(numbers):
    out, total = [], 0
    for n in numbers:
        total += n
        out.append(total)
    return out
""",
    "merge_sorted": """
def merge_sorted(a, b):
    return sorted(list(a) + list(b))
""",
    "balanced_brackets": """
def balanced_brackets(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
""",
    "longest_common_prefix": """
def longest_common_prefix(strings):
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
""",
    "binary_search": """
def binary_search(values, target):
    lo, hi = 0, len(values) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if values[mid] == target:
            return mid
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
    "collatz_steps": """
def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps
""",
    "primes_up_to": """
def primes_up_to(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]
""",
    "compress_string": """
def compress_string(text):
    if not text:
        return text
    parts, run_char, run_len = [], text[0], 1
    for ch in text[1:]:
        if ch == run_char:
            run_len += 1
        else:
            parts.append(run_char + str(run_len))
            run_char, run_len = ch, 1
    parts.append(run_char + str(run_len))
    encoded = "".join(parts)
    return encoded if len(encoded) < len(text) else text
""",
    "chunk_list": """
def chunk_list(items, size):
    if size <= 0:
        return []
    return [items[i:i + size] for i in range(0, len(items), size)]
""",
    "caesar_cipher": """
def caesar_cipher(text, shift):
    out = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            out.append(chr(base + (ord(ch) - base + shift) % 26))
        else:
            out.append(ch)
    return "".join(out)
""",
    "valid_ipv4": """
def valid_ipv4(text):
    parts = text.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        if len(part) > 1 and part[0] == "0":
            return False
        if int(part) > 255:
            return False
    return True
""",
    "digital_root": """
def digital_root(n):
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n
""",
    "find_duplicates": """
def find_duplicates(items):
    seen, dupes = set(), set()
    for item in items:
        if item in seen:
            dupes.add(item)
        seen.add(item)
    return sorted(dupes)
""",
    "max_subarray_sum": """
def max_subarray_sum(numbers):
    best = current = numbers[0]
    for n in numbers[1:]:
        current = max(n, current + n)
        best = max(best, current)
    return best
""",
    "moving_average": """
def moving_average(numbers, window):
    if window > len(numbers):
        return []
    return [round(sum(numbers[i:i + window]) / window, 4)
            for i in range(len(numbers) - window + 1)]
""",
    "word_frequencies": """
def word_frequencies(text):
    counts = {}
    for word in text.split():
        cleaned = word.strip(".,!?;:").lower()
        if cleaned:
            counts[cleaned] = counts.get(cleaned, 0) + 1
    return counts
""",
    "matrix_transpose": """
def matrix_transpose(matrix):
    return [list(row) for row in zip(*matrix)]
""",
    "intersect_sorted": """
def intersect_sorted(a, b):
    return sorted(set(a) & set(b))
""",
    "longest_increasing_run": """
def longest_increasing_run(numbers):
    if not numbers:
        return 0
    best = run = 1
    for prev, cur in zip(numbers, numbers[1:]):
        run = run + 1 if cur > prev else 1
        best = max(best, run)
    return best
""",
    "roman_to_int": """
def roman_to_int(text):
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(text):
        value = values[ch]
        if i + 1 < len(text) and value < values[text[i + 1]]:
            total -= value
        else:
            total += value
    return total
""",
    "int_to_roman": """
def int_to_roman(n):
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
             (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
             (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for value, symbol in table:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)
""",
    "group_anagrams": """
def group_anagrams(words):
    groups = {}
    for word in words:
        key = "".join(sorted(word))
        groups.setdefault(key, []).append(word)
    return {key: sorted(members) for key, members in groups.items()}
""",
    "two_sum": """
def two_sum(numbers, target):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                return [i, j]
    return []
""",
    "spiral_order": """
def spiral_order(matrix):
    if not matrix:
        return []
    rows = [list(row) for row in matrix]
    out = []
    while rows:
        out.extend(rows.pop(0))
        rows = [list(row) for row in zip(*rows)][::-1]
    return out
""",
    "rotate_matrix": """
def rotate_matrix(matrix):
    return [list(row) for row in zip(*matrix[::-1])]
""",
    "gcd_list": """
def gcd_list(numbers):
    from math import gcd
    result = numbers[0]
    for n in numbers[1:]:
        result = gcd(result, n)
    return result
""",
    "count_islands": """
def count_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = set()
    islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 1 or (r, c) in seen:
                continue
            islands += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr+1, cc), (cr-1, cc), (cr, cc+1), (cr, cc-1)):
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr][nc] == 1 and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
    return islands
""",
    "is_anagram": """
def is_anagram(a, b):
    return sorted(a.lower().replace(" ", "")) == sorted(b.lower().replace(" ", ""))
""",
    "parse_query_string": """
def parse_query_string(text):
    result = {}
    for piece in text.split("&"):
        if not piece:
            continue
        key, _, value = piece.partition("=")
        result[key] = value
    return result
""",
    "most_common_word": """
def most_common_word(text):
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    if not counts:
        return ""
    return min(counts, key=lambda w: (-counts[w], w))
""",
    "remove_nth_from_end": """
def remove_nth_from_end(items, n):
    out = list(items)
    if 1 <= n <= len(out):
        del out[len(out) - n]
    return out
""",
}


def local_subprocess_runner(source: str, timeout: float) -> CodeRunResult:
    proc = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CodeRunResult(proc.stdout, proc.stderr, proc.returncode, timed_out=False)


ALL_TASKS = generate_code_tasks("train") + generate_code_tasks("heldout")


def test_every_problem_has_a_reference_solution():
    assert {p.name for p in PROBLEMS} == set(REFERENCE)


def test_problem_names_and_entry_points_are_unique():
    names = [p.name for p in PROBLEMS]
    assert len(set(names)) == len(names)
    assert len({p.entry_point for p in PROBLEMS}) == len(names)


def test_every_problem_has_enough_test_cases():
    """One test case is a coin flip, not a verifier."""
    for problem in PROBLEMS:
        assert len(problem.tests) >= 3, problem.name


def test_held_out_problems_are_distinct_problems():
    train = {p.name for p in PROBLEMS if p.split == "train"}
    heldout = {p.name for p in PROBLEMS if p.split == "heldout"}
    assert train and heldout
    assert train.isdisjoint(heldout)


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.metadata["name"])
def test_reference_solution_scores_one(task):
    source = REFERENCE[task.metadata["name"]]
    response = f"<answer>\n```python\n{source}\n```\n</answer>"
    assert verify(task, response, code_runner=local_subprocess_runner) == 1.0


@pytest.mark.parametrize("task", ALL_TASKS[:6], ids=lambda t: t.metadata["name"])
def test_an_empty_function_does_not_score(task):
    """A stub that returns None must not accidentally satisfy the tests."""
    stub = f"def {task.ground_truth['entry_point']}(*args, **kwargs):\n    return None\n"
    response = f"<answer>\n```python\n{stub}\n```\n</answer>"
    assert verify(task, response, code_runner=local_subprocess_runner) < 1.0
