"""Hand-authored coding problems, checked by running the generated code.

Unlike the math set these cannot be generated from templates — a coding
problem's difficulty lives in its specification, not in its numbers, and
thirty instances of one templated problem would measure one skill thirty
times. So each problem is written out, with its test cases written by hand
alongside it.

Hand-written expected values are exactly where typos hide, so
`tests/test_tasks_code.py` holds an independently written reference solution
for every problem and asserts it passes that problem's tests through the
real verifier. Two independent derivations agreeing is what makes these
expectations trustworthy; a reference solution alone would only prove the
code equals itself.

Specifications state their edge cases explicitly (what an empty input
returns, how ties break) because scoring is exact: an unstated convention is
a test the policy cannot pass by reasoning, only by guessing.

Held-out coding tasks are different *problems*, not different instances of
the same problem — the only way a held-out coding score means anything.

Test arguments and expected values stay JSON-native; see verify_code.
"""

from dataclasses import dataclass

from .schema import Task

PROMPT_SUFFIX = (
    " Return the complete function definition; you may define helper functions "
    "alongside it."
)


@dataclass(frozen=True)
class CodeProblem:
    name: str
    split: str
    entry_point: str
    prompt: str
    tests: list[dict]


PROBLEMS: tuple[CodeProblem, ...] = (
    # ---------------------------------------------------------------- train
    CodeProblem(
        name="sum_of_evens",
        split="train",
        entry_point="sum_of_evens",
        prompt=(
            "Write a Python function `sum_of_evens(numbers)` that takes a list of "
            "integers and returns the sum of the even ones. An empty list returns 0."
        ),
        tests=[
            {"args": [[1, 2, 3, 4]], "expected": 6},
            {"args": [[]], "expected": 0},
            {"args": [[1, 3, 5]], "expected": 0},
            {"args": [[-2, -3, 4]], "expected": 2},
            {"args": [[0]], "expected": 0},
        ],
    ),
    CodeProblem(
        name="count_vowels",
        split="train",
        entry_point="count_vowels",
        prompt=(
            "Write a Python function `count_vowels(text)` that returns how many "
            "vowels the string contains. The vowels are a, e, i, o and u, in either "
            "case; y does not count."
        ),
        tests=[
            {"args": ["hello"], "expected": 2},
            {"args": ["AEIOU"], "expected": 5},
            {"args": ["rhythm"], "expected": 0},
            {"args": [""], "expected": 0},
            {"args": ["Programming"], "expected": 3},
        ],
    ),
    CodeProblem(
        name="reverse_words",
        split="train",
        entry_point="reverse_words",
        prompt=(
            "Write a Python function `reverse_words(sentence)` that reverses the "
            "order of the words in a string. Words are separated by any amount of "
            "whitespace, and the result joins them with a single space, with no "
            "leading or trailing space."
        ),
        tests=[
            {"args": ["the quick brown fox"], "expected": "fox brown quick the"},
            {"args": ["hello"], "expected": "hello"},
            {"args": [""], "expected": ""},
            {"args": ["  a  b  "], "expected": "b a"},
        ],
    ),
    CodeProblem(
        name="is_palindrome",
        split="train",
        entry_point="is_palindrome",
        prompt=(
            "Write a Python function `is_palindrome(text)` that returns True if the "
            "string reads the same forwards and backwards, ignoring case and any "
            "character that is not a letter or digit. An empty string is a "
            "palindrome."
        ),
        tests=[
            {"args": ["A man, a plan, a canal: Panama"], "expected": True},
            {"args": ["hello"], "expected": False},
            {"args": [""], "expected": True},
            {"args": ["race a car"], "expected": False},
            {"args": ["No 'x' in Nixon"], "expected": True},
        ],
    ),
    CodeProblem(
        name="second_largest",
        split="train",
        entry_point="second_largest",
        prompt=(
            "Write a Python function `second_largest(numbers)` that returns the "
            "second largest *distinct* value in a list of integers, or None if the "
            "list has fewer than two distinct values."
        ),
        tests=[
            {"args": [[3, 1, 4, 1, 5]], "expected": 4},
            {"args": [[2, 2, 2]], "expected": None},
            {"args": [[1]], "expected": None},
            {"args": [[]], "expected": None},
            {"args": [[-5, -2, -9]], "expected": -5},
        ],
    ),
    CodeProblem(
        name="running_total",
        split="train",
        entry_point="running_total",
        prompt=(
            "Write a Python function `running_total(numbers)` that returns a list of "
            "the cumulative sums of a list of integers, so that element i of the "
            "result is the sum of elements 0 through i of the input. An empty list "
            "returns an empty list."
        ),
        tests=[
            {"args": [[1, 2, 3]], "expected": [1, 3, 6]},
            {"args": [[]], "expected": []},
            {"args": [[5]], "expected": [5]},
            {"args": [[1, -1, 2]], "expected": [1, 0, 2]},
        ],
    ),
    CodeProblem(
        name="merge_sorted",
        split="train",
        entry_point="merge_sorted",
        prompt=(
            "Write a Python function `merge_sorted(a, b)` that merges two lists of "
            "integers, each already sorted in ascending order, into one sorted list. "
            "Duplicates are kept."
        ),
        tests=[
            {"args": [[1, 3, 5], [2, 4]], "expected": [1, 2, 3, 4, 5]},
            {"args": [[], [1]], "expected": [1]},
            {"args": [[1, 1], [1]], "expected": [1, 1, 1]},
            {"args": [[], []], "expected": []},
        ],
    ),
    CodeProblem(
        name="balanced_brackets",
        split="train",
        entry_point="balanced_brackets",
        prompt=(
            "Write a Python function `balanced_brackets(text)` that returns True if "
            "every (, [ and { in the string is closed by the matching bracket in the "
            "correct order. Any other character is ignored."
        ),
        tests=[
            {"args": ["()"], "expected": True},
            {"args": ["([{}])"], "expected": True},
            {"args": ["(]"], "expected": False},
            {"args": ["(("], "expected": False},
            {"args": ["a(b)c"], "expected": True},
            {"args": [")("], "expected": False},
        ],
    ),
    CodeProblem(
        name="longest_common_prefix",
        split="train",
        entry_point="longest_common_prefix",
        prompt=(
            "Write a Python function `longest_common_prefix(strings)` that returns "
            "the longest string that every string in the list starts with. Return an "
            "empty string if there is no common prefix or the list is empty."
        ),
        tests=[
            {"args": [["flower", "flow", "flight"]], "expected": "fl"},
            {"args": [["dog", "racecar"]], "expected": ""},
            {"args": [["abc"]], "expected": "abc"},
            {"args": [[]], "expected": ""},
            {"args": [["", ""]], "expected": ""},
        ],
    ),
    CodeProblem(
        name="binary_search",
        split="train",
        entry_point="binary_search",
        prompt=(
            "Write a Python function `binary_search(values, target)` that returns the "
            "index of `target` in `values`, a list of distinct integers sorted in "
            "ascending order, or -1 if it is not present."
        ),
        tests=[
            {"args": [[1, 3, 5, 7, 9], 7], "expected": 3},
            {"args": [[1, 3, 5, 7, 9], 4], "expected": -1},
            {"args": [[], 1], "expected": -1},
            {"args": [[2], 2], "expected": 0},
        ],
    ),
    CodeProblem(
        name="collatz_steps",
        split="train",
        entry_point="collatz_steps",
        prompt=(
            "Write a Python function `collatz_steps(n)` that returns how many steps "
            "the Collatz sequence takes to reach 1 from a positive integer n. Each "
            "step halves an even number or replaces an odd number n with 3n + 1. "
            "`collatz_steps(1)` is 0."
        ),
        tests=[
            {"args": [1], "expected": 0},
            {"args": [2], "expected": 1},
            {"args": [6], "expected": 8},
            {"args": [27], "expected": 111},
        ],
    ),
    CodeProblem(
        name="primes_up_to",
        split="train",
        entry_point="primes_up_to",
        prompt=(
            "Write a Python function `primes_up_to(n)` that returns a list, in "
            "ascending order, of every prime number less than or equal to n."
        ),
        tests=[
            {"args": [10], "expected": [2, 3, 5, 7]},
            {"args": [1], "expected": []},
            {"args": [2], "expected": [2]},
            {"args": [20], "expected": [2, 3, 5, 7, 11, 13, 17, 19]},
            {"args": [0], "expected": []},
        ],
    ),
    CodeProblem(
        name="compress_string",
        split="train",
        entry_point="compress_string",
        prompt=(
            "Write a Python function `compress_string(text)` that run-length encodes "
            "a string, so that 'aaabb' becomes 'a3b2'. Every run is written as the "
            "character followed by its count, including runs of length 1. If the "
            "encoded string is not strictly shorter than the original, return the "
            "original unchanged."
        ),
        tests=[
            {"args": ["aaabb"], "expected": "a3b2"},
            {"args": ["abc"], "expected": "abc"},
            {"args": [""], "expected": ""},
            {"args": ["aabbccc"], "expected": "a2b2c3"},
            {"args": ["aa"], "expected": "aa"},
        ],
    ),
    CodeProblem(
        name="chunk_list",
        split="train",
        entry_point="chunk_list",
        prompt=(
            "Write a Python function `chunk_list(items, size)` that splits a list "
            "into consecutive chunks of at most `size` elements. The last chunk may "
            "be shorter. Return an empty list if `size` is zero or negative."
        ),
        tests=[
            {"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]},
            {"args": [[], 3], "expected": []},
            {"args": [[1, 2], 5], "expected": [[1, 2]]},
            {"args": [[1, 2, 3], 0], "expected": []},
        ],
    ),
    CodeProblem(
        name="caesar_cipher",
        split="train",
        entry_point="caesar_cipher",
        prompt=(
            "Write a Python function `caesar_cipher(text, shift)` that shifts every "
            "letter in the string forward by `shift` places through the alphabet, "
            "wrapping around from z to a. Case is preserved and non-letters are left "
            "alone. A negative shift moves backwards."
        ),
        tests=[
            {"args": ["abc", 1], "expected": "bcd"},
            {"args": ["xyz", 3], "expected": "abc"},
            {"args": ["Hello, World!", 5], "expected": "Mjqqt, Btwqi!"},
            {"args": ["abc", -1], "expected": "zab"},
            {"args": ["abc", 0], "expected": "abc"},
        ],
    ),
    CodeProblem(
        name="valid_ipv4",
        split="train",
        entry_point="valid_ipv4",
        prompt=(
            "Write a Python function `valid_ipv4(text)` that returns True if the "
            "string is a valid IPv4 address: exactly four parts separated by dots, "
            "each a decimal number from 0 to 255 with no leading zeros (so '0' is "
            "allowed but '01' is not)."
        ),
        tests=[
            {"args": ["192.168.0.1"], "expected": True},
            {"args": ["255.255.255.255"], "expected": True},
            {"args": ["256.1.1.1"], "expected": False},
            {"args": ["1.1.1"], "expected": False},
            {"args": ["01.1.1.1"], "expected": False},
            {"args": ["1.1.1.1.1"], "expected": False},
            {"args": [""], "expected": False},
        ],
    ),
    CodeProblem(
        name="digital_root",
        split="train",
        entry_point="digital_root",
        prompt=(
            "Write a Python function `digital_root(n)` that repeatedly sums the "
            "digits of a non-negative integer until one digit is left, and returns "
            "that digit."
        ),
        tests=[
            {"args": [0], "expected": 0},
            {"args": [9], "expected": 9},
            {"args": [38], "expected": 2},
            {"args": [12345], "expected": 6},
            {"args": [100], "expected": 1},
        ],
    ),
    CodeProblem(
        name="find_duplicates",
        split="train",
        entry_point="find_duplicates",
        prompt=(
            "Write a Python function `find_duplicates(items)` that returns, in "
            "ascending order, every integer that appears more than once in the list. "
            "Each such value appears once in the result."
        ),
        tests=[
            {"args": [[1, 2, 2, 3, 3, 3]], "expected": [2, 3]},
            {"args": [[1, 2, 3]], "expected": []},
            {"args": [[]], "expected": []},
            {"args": [[5, 5, 1, 1]], "expected": [1, 5]},
        ],
    ),
    CodeProblem(
        name="max_subarray_sum",
        split="train",
        entry_point="max_subarray_sum",
        prompt=(
            "Write a Python function `max_subarray_sum(numbers)` that returns the "
            "largest sum obtainable from any non-empty contiguous slice of a "
            "non-empty list of integers."
        ),
        tests=[
            {"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"args": [[1]], "expected": 1},
            {"args": [[-3, -1, -2]], "expected": -1},
            {"args": [[2, 3]], "expected": 5},
        ],
    ),
    CodeProblem(
        name="moving_average",
        split="train",
        entry_point="moving_average",
        prompt=(
            "Write a Python function `moving_average(numbers, window)` that returns "
            "the average of every consecutive group of `window` values, each rounded "
            "to four decimal places. The result has len(numbers) - window + 1 "
            "entries; return an empty list if `window` is larger than the list."
        ),
        tests=[
            {"args": [[1, 2, 3, 4], 2], "expected": [1.5, 2.5, 3.5]},
            {"args": [[1, 2, 3], 5], "expected": []},
            {"args": [[1, 2, 3], 1], "expected": [1.0, 2.0, 3.0]},
            {"args": [[1, 2, 3, 4, 5], 3], "expected": [2.0, 3.0, 4.0]},
        ],
    ),
    CodeProblem(
        name="word_frequencies",
        split="train",
        entry_point="word_frequencies",
        prompt=(
            "Write a Python function `word_frequencies(text)` that returns a "
            "dictionary mapping each word in the string to how many times it "
            "appears. Words are separated by whitespace, compared in lower case, and "
            "stripped of any leading or trailing characters from .,!?;: — an empty "
            "string returns an empty dictionary."
        ),
        tests=[
            {"args": ["the cat the dog"], "expected": {"the": 2, "cat": 1, "dog": 1}},
            {"args": ["Hello, hello world!"], "expected": {"hello": 2, "world": 1}},
            {"args": [""], "expected": {}},
        ],
    ),
    CodeProblem(
        name="matrix_transpose",
        split="train",
        entry_point="matrix_transpose",
        prompt=(
            "Write a Python function `matrix_transpose(matrix)` that returns the "
            "transpose of a rectangular matrix given as a list of equal-length rows. "
            "An empty matrix transposes to an empty list."
        ),
        tests=[
            {"args": [[[1, 2], [3, 4]]], "expected": [[1, 3], [2, 4]]},
            {"args": [[[1, 2, 3]]], "expected": [[1], [2], [3]]},
            {"args": [[]], "expected": []},
        ],
    ),
    CodeProblem(
        name="intersect_sorted",
        split="train",
        entry_point="intersect_sorted",
        prompt=(
            "Write a Python function `intersect_sorted(a, b)` that returns, in "
            "ascending order and without repeats, every value that appears in both "
            "lists of integers."
        ),
        tests=[
            {"args": [[1, 2, 3, 4], [2, 4, 6]], "expected": [2, 4]},
            {"args": [[1, 1, 2], [1]], "expected": [1]},
            {"args": [[], [1]], "expected": []},
            {"args": [[1], [2]], "expected": []},
        ],
    ),
    CodeProblem(
        name="longest_increasing_run",
        split="train",
        entry_point="longest_increasing_run",
        prompt=(
            "Write a Python function `longest_increasing_run(numbers)` that returns "
            "the length of the longest strictly increasing run of consecutive "
            "elements in the list. An empty list returns 0."
        ),
        tests=[
            {"args": [[1, 2, 3, 1, 2]], "expected": 3},
            {"args": [[]], "expected": 0},
            {"args": [[5, 4, 3]], "expected": 1},
            {"args": [[1, 2, 2, 3]], "expected": 2},
            {"args": [[1]], "expected": 1},
        ],
    ),
    CodeProblem(
        name="title_case_words",
        split="train",
        entry_point="title_case_words",
        prompt=(
            "Write a Python function `title_case_words(text)` that capitalises the "
            "first letter of every whitespace-separated word and lower-cases the "
            "rest, joining the words with a single space and no leading or trailing "
            "space."
        ),
        tests=[
            {"args": ["hello world"], "expected": "Hello World"},
            {"args": ["HELLO wORLD"], "expected": "Hello World"},
            {"args": [""], "expected": ""},
            {"args": ["  a  b "], "expected": "A B"},
            {"args": ["python3 rocks"], "expected": "Python3 Rocks"},
        ],
    ),
    CodeProblem(
        name="count_substring",
        split="train",
        entry_point="count_substring",
        prompt=(
            "Write a Python function `count_substring(text, needle)` that counts how "
            "many times `needle` occurs in `text` without overlapping, scanning left "
            "to right. An empty needle returns 0."
        ),
        tests=[
            {"args": ["banana", "an"], "expected": 2},
            {"args": ["aaaa", "aa"], "expected": 2},
            {"args": ["abc", "d"], "expected": 0},
            {"args": ["abc", ""], "expected": 0},
            {"args": ["aaa", "a"], "expected": 3},
        ],
    ),
    CodeProblem(
        name="sum_of_digits_in_string",
        split="train",
        entry_point="sum_of_digits_in_string",
        prompt=(
            "Write a Python function `sum_of_digits_in_string(text)` that adds up "
            "every digit character in the string and returns the total, ignoring "
            "every other character. A string with no digits returns 0."
        ),
        tests=[
            {"args": ["a1b2c3"], "expected": 6},
            {"args": [""], "expected": 0},
            {"args": ["no digits"], "expected": 0},
            {"args": ["2024"], "expected": 8},
            {"args": ["-5"], "expected": 5},
        ],
    ),
    CodeProblem(
        name="remove_duplicates_preserving_order",
        split="train",
        entry_point="remove_duplicates_preserving_order",
        prompt=(
            "Write a Python function `remove_duplicates_preserving_order(items)` that "
            "returns a list keeping only the first occurrence of each integer, in the "
            "order those first occurrences appeared."
        ),
        tests=[
            {"args": [[1, 2, 1, 3, 2]], "expected": [1, 2, 3]},
            {"args": [[]], "expected": []},
            {"args": [[1, 1, 1]], "expected": [1]},
            {"args": [[3, 2, 1]], "expected": [3, 2, 1]},
        ],
    ),
    CodeProblem(
        name="is_perfect_square",
        split="train",
        entry_point="is_perfect_square",
        prompt=(
            "Write a Python function `is_perfect_square(n)` that returns True if the "
            "non-negative integer n is the square of an integer. 0 is a perfect "
            "square."
        ),
        tests=[
            {"args": [0], "expected": True},
            {"args": [1], "expected": True},
            {"args": [2], "expected": False},
            {"args": [15], "expected": False},
            {"args": [16], "expected": True},
            {"args": [10000], "expected": True},
        ],
    ),
    CodeProblem(
        name="nth_fibonacci",
        split="train",
        entry_point="nth_fibonacci",
        prompt=(
            "Write a Python function `nth_fibonacci(n)` that returns the nth "
            "Fibonacci number, counting from 0, where the sequence starts 0, 1, 1, "
            "2, 3, 5."
        ),
        tests=[
            {"args": [0], "expected": 0},
            {"args": [1], "expected": 1},
            {"args": [2], "expected": 1},
            {"args": [10], "expected": 55},
            {"args": [30], "expected": 832040},
        ],
    ),
    # -------------------------------------------------------------- heldout
    CodeProblem(
        name="roman_to_int",
        split="heldout",
        entry_point="roman_to_int",
        prompt=(
            "Write a Python function `roman_to_int(text)` that converts a valid Roman "
            "numeral string to an integer, handling subtractive pairs such as IV, IX, "
            "XL, XC, CD and CM."
        ),
        tests=[
            {"args": ["III"], "expected": 3},
            {"args": ["IV"], "expected": 4},
            {"args": ["IX"], "expected": 9},
            {"args": ["LVIII"], "expected": 58},
            {"args": ["MCMXCIV"], "expected": 1994},
        ],
    ),
    CodeProblem(
        name="int_to_roman",
        split="heldout",
        entry_point="int_to_roman",
        prompt=(
            "Write a Python function `int_to_roman(n)` that converts an integer "
            "between 1 and 3999 to its Roman numeral string, using the subtractive "
            "forms IV, IX, XL, XC, CD and CM."
        ),
        tests=[
            {"args": [3], "expected": "III"},
            {"args": [4], "expected": "IV"},
            {"args": [9], "expected": "IX"},
            {"args": [58], "expected": "LVIII"},
            {"args": [1994], "expected": "MCMXCIV"},
            {"args": [3999], "expected": "MMMCMXCIX"},
        ],
    ),
    CodeProblem(
        name="group_anagrams",
        split="heldout",
        entry_point="group_anagrams",
        prompt=(
            "Write a Python function `group_anagrams(words)` that groups words that "
            "are anagrams of one another. Return a dictionary whose key is the word's "
            "letters sorted into alphabetical order and whose value is the list of "
            "matching words sorted alphabetically."
        ),
        tests=[
            {
                "args": [["eat", "tea", "tan", "ate", "nat", "bat"]],
                "expected": {"aet": ["ate", "eat", "tea"], "ant": ["nat", "tan"], "abt": ["bat"]},
            },
            {"args": [[]], "expected": {}},
            {"args": [["abc"]], "expected": {"abc": ["abc"]}},
        ],
    ),
    CodeProblem(
        name="two_sum",
        split="heldout",
        entry_point="two_sum",
        prompt=(
            "Write a Python function `two_sum(numbers, target)` that returns the two "
            "indices of values summing to `target`, as a list [i, j] with i < j. If "
            "several pairs work, return the one with the smallest i, and among those "
            "the smallest j. Return an empty list if no pair sums to the target."
        ),
        tests=[
            {"args": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"args": [[3, 2, 4], 6], "expected": [1, 2]},
            {"args": [[3, 3], 6], "expected": [0, 1]},
            {"args": [[1, 2], 10], "expected": []},
            {"args": [[0, 4, 3, 0], 0], "expected": [0, 3]},
        ],
    ),
    CodeProblem(
        name="spiral_order",
        split="heldout",
        entry_point="spiral_order",
        prompt=(
            "Write a Python function `spiral_order(matrix)` that returns every "
            "element of a rectangular matrix in clockwise spiral order, starting at "
            "the top-left and moving right. An empty matrix returns an empty list."
        ),
        tests=[
            {
                "args": [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                "expected": [1, 2, 3, 6, 9, 8, 7, 4, 5],
            },
            {"args": [[[1, 2], [3, 4]]], "expected": [1, 2, 4, 3]},
            {"args": [[[1]]], "expected": [1]},
            {"args": [[]], "expected": []},
        ],
    ),
    CodeProblem(
        name="rotate_matrix",
        split="heldout",
        entry_point="rotate_matrix",
        prompt=(
            "Write a Python function `rotate_matrix(matrix)` that returns a new "
            "square matrix rotated 90 degrees clockwise, so the first row of the "
            "result is the first column of the input read from the bottom up."
        ),
        tests=[
            {"args": [[[1, 2], [3, 4]]], "expected": [[3, 1], [4, 2]]},
            {
                "args": [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                "expected": [[7, 4, 1], [8, 5, 2], [9, 6, 3]],
            },
            {"args": [[[1]]], "expected": [[1]]},
            {"args": [[]], "expected": []},
        ],
    ),
    CodeProblem(
        name="gcd_list",
        split="heldout",
        entry_point="gcd_list",
        prompt=(
            "Write a Python function `gcd_list(numbers)` that returns the greatest "
            "common divisor of a non-empty list of positive integers. A list with one "
            "element returns that element."
        ),
        tests=[
            {"args": [[12, 18]], "expected": 6},
            {"args": [[7]], "expected": 7},
            {"args": [[4, 8, 16]], "expected": 4},
            {"args": [[9, 28]], "expected": 1},
        ],
    ),
    CodeProblem(
        name="count_islands",
        split="heldout",
        entry_point="count_islands",
        prompt=(
            "Write a Python function `count_islands(grid)` that counts the connected "
            "groups of 1s in a rectangular grid of 0s and 1s. Cells are connected "
            "only horizontally or vertically, not diagonally. An empty grid has 0 "
            "islands."
        ),
        tests=[
            {"args": [[[1, 1, 0], [0, 1, 0], [1, 0, 1]]], "expected": 3},
            {"args": [[[0, 0], [0, 0]]], "expected": 0},
            {"args": [[[1]]], "expected": 1},
            {"args": [[]], "expected": 0},
        ],
    ),
    CodeProblem(
        name="is_anagram",
        split="heldout",
        entry_point="is_anagram",
        prompt=(
            "Write a Python function `is_anagram(a, b)` that returns True if the two "
            "strings contain the same letters in the same quantities, ignoring case "
            "and spaces."
        ),
        tests=[
            {"args": ["listen", "silent"], "expected": True},
            {"args": ["Dormitory", "dirty room"], "expected": True},
            {"args": ["abc", "abd"], "expected": False},
            {"args": ["", ""], "expected": True},
        ],
    ),
    CodeProblem(
        name="parse_query_string",
        split="heldout",
        entry_point="parse_query_string",
        prompt=(
            "Write a Python function `parse_query_string(text)` that parses a URL "
            "query string such as 'a=1&b=2' into a dictionary of string keys to "
            "string values. A key with nothing after the '=' maps to an empty string, "
            "a repeated key keeps the last value, and an empty input returns an empty "
            "dictionary."
        ),
        tests=[
            {"args": ["a=1&b=2"], "expected": {"a": "1", "b": "2"}},
            {"args": [""], "expected": {}},
            {"args": ["x=1&x=2"], "expected": {"x": "2"}},
            {"args": ["flag="], "expected": {"flag": ""}},
        ],
    ),
    CodeProblem(
        name="most_common_word",
        split="heldout",
        entry_point="most_common_word",
        prompt=(
            "Write a Python function `most_common_word(text)` that returns the most "
            "frequent word in the string, compared in lower case and split on "
            "whitespace. If several words tie, return the one that comes first "
            "alphabetically. An empty string returns an empty string."
        ),
        tests=[
            {"args": ["the cat the dog"], "expected": "the"},
            {"args": ["a b"], "expected": "a"},
            {"args": [""], "expected": ""},
            {"args": ["Bob bob alice"], "expected": "bob"},
        ],
    ),
    CodeProblem(
        name="remove_nth_from_end",
        split="heldout",
        entry_point="remove_nth_from_end",
        prompt=(
            "Write a Python function `remove_nth_from_end(items, n)` that returns a "
            "new list with the nth element from the end removed, where n = 1 is the "
            "last element. If n is outside the list, return an unchanged copy."
        ),
        tests=[
            {"args": [[1, 2, 3, 4, 5], 2], "expected": [1, 2, 3, 5]},
            {"args": [[1], 1], "expected": []},
            {"args": [[1, 2, 3], 5], "expected": [1, 2, 3]},
            {"args": [[1, 2, 3], 0], "expected": [1, 2, 3]},
        ],
    ),
    CodeProblem(
        name="longest_word",
        split="heldout",
        entry_point="longest_word",
        prompt=(
            "Write a Python function `longest_word(text)` that returns the longest "
            "whitespace-separated word in the string. If several words tie for "
            "longest, return the one that appears first. An empty string returns an "
            "empty string."
        ),
        tests=[
            {"args": ["the quick brown fox"], "expected": "quick"},
            {"args": [""], "expected": ""},
            {"args": ["a bb ccc"], "expected": "ccc"},
            {"args": ["aa bb"], "expected": "aa"},
        ],
    ),
    CodeProblem(
        name="rotate_list",
        split="heldout",
        entry_point="rotate_list",
        prompt=(
            "Write a Python function `rotate_list(items, k)` that returns a new list "
            "rotated k places to the right, so the last k elements move to the front. "
            "k may be larger than the list or negative, where a negative k rotates "
            "left. An empty list returns an empty list."
        ),
        tests=[
            {"args": [[1, 2, 3, 4, 5], 2], "expected": [4, 5, 1, 2, 3]},
            {"args": [[1, 2, 3], 0], "expected": [1, 2, 3]},
            {"args": [[], 3], "expected": []},
            {"args": [[1, 2, 3], 4], "expected": [3, 1, 2]},
            {"args": [[1, 2, 3], -1], "expected": [2, 3, 1]},
        ],
    ),
    CodeProblem(
        name="sum_nested",
        split="heldout",
        entry_point="sum_nested",
        prompt=(
            "Write a Python function `sum_nested(nested)` that returns the sum of "
            "every integer inside a list that may contain further lists, nested to "
            "any depth. A list containing no integers sums to 0."
        ),
        tests=[
            {"args": [[1, [2, 3], [[4]]]], "expected": 10},
            {"args": [[]], "expected": 0},
            {"args": [[[[]]]], "expected": 0},
            {"args": [[1, 2, 3]], "expected": 6},
        ],
    ),
    CodeProblem(
        name="max_nesting_depth",
        split="heldout",
        entry_point="max_nesting_depth",
        prompt=(
            "Write a Python function `max_nesting_depth(text)` that returns how "
            "deeply the round brackets in the string are nested at their deepest "
            "point. The brackets are guaranteed to be balanced; a string with none "
            "returns 0."
        ),
        tests=[
            {"args": [""], "expected": 0},
            {"args": ["()"], "expected": 1},
            {"args": ["(())"], "expected": 2},
            {"args": ["()()"], "expected": 1},
            {"args": ["(()(()))"], "expected": 3},
        ],
    ),
    CodeProblem(
        name="pascals_triangle_row",
        split="heldout",
        entry_point="pascals_triangle_row",
        prompt=(
            "Write a Python function `pascals_triangle_row(n)` that returns row n of "
            "Pascal's triangle as a list, counting from 0, so row 0 is [1] and row 1 "
            "is [1, 1]."
        ),
        tests=[
            {"args": [0], "expected": [1]},
            {"args": [1], "expected": [1, 1]},
            {"args": [2], "expected": [1, 2, 1]},
            {"args": [4], "expected": [1, 4, 6, 4, 1]},
        ],
    ),
    CodeProblem(
        name="invert_dict",
        split="heldout",
        entry_point="invert_dict",
        prompt=(
            "Write a Python function `invert_dict(mapping)` that returns a new "
            "dictionary with the keys and values of a dictionary of strings swapped. "
            "The values are guaranteed to be unique. An empty dictionary returns an "
            "empty dictionary."
        ),
        tests=[
            {"args": [{"a": "1", "b": "2"}], "expected": {"1": "a", "2": "b"}},
            {"args": [{}], "expected": {}},
            {"args": [{"x": "y"}], "expected": {"y": "x"}},
        ],
    ),
    CodeProblem(
        name="merge_intervals",
        split="heldout",
        entry_point="merge_intervals",
        prompt=(
            "Write a Python function `merge_intervals(intervals)` that merges "
            "overlapping intervals, each given as a [start, end] pair, and returns "
            "the merged list sorted by start. Intervals that merely touch at an "
            "endpoint count as overlapping."
        ),
        tests=[
            {"args": [[[1, 3], [2, 6], [8, 10]]], "expected": [[1, 6], [8, 10]]},
            {"args": [[]], "expected": []},
            {"args": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
            {"args": [[[5, 6], [1, 2]]], "expected": [[1, 2], [5, 6]]},
        ],
    ),
    CodeProblem(
        name="first_non_repeating_char",
        split="heldout",
        entry_point="first_non_repeating_char",
        prompt=(
            "Write a Python function `first_non_repeating_char(text)` that returns "
            "the first character occurring exactly once in the string. If every "
            "character repeats, or the string is empty, return an empty string."
        ),
        tests=[
            {"args": ["swiss"], "expected": "w"},
            {"args": ["aabb"], "expected": ""},
            {"args": [""], "expected": ""},
            {"args": ["abcabd"], "expected": "c"},
        ],
    ),
)


def generate_code_tasks(split: str) -> list[Task]:
    """Return every authored coding problem belonging to `split`."""
    chosen = [p for p in PROBLEMS if p.split == split]
    return [
        Task(
            task_id=f"code-{split}-{i:04d}",
            category="code",
            prompt=problem.prompt + PROMPT_SUFFIX,
            ground_truth={"entry_point": problem.entry_point, "tests": problem.tests},
            split=split,
            metadata={"name": problem.name, "n_tests": len(problem.tests)},
        )
        for i, problem in enumerate(chosen)
    ]
