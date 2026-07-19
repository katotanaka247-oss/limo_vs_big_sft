"""
math_utils.py
本地数学评测任务的评分函数，从 lm-evaluation-harness v0.4.5 的
lm_eval/tasks/hendrycks_math/utils.py 移植。

支持判定：
  * \\boxed{} 答案抽取；
  * 整数、分数、小数、LaTeX 表达式；
  * AIME 的 0-999 整数答案；
  * MATH500 的符号表达式（经 strip_string 归一化后比较）。

判定方式：数学等价归一化（is_equiv / strip_string），而非简单字符串相等。

来源记录见 eval_tasks/README.md。
"""
from typing import Dict, List


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """lm-eval 评分入口：doc_to_target 已在 yaml 中指向 answer 字段，
    这里同时做 \\boxed 抽取 + 等价判定。"""
    retval = 0
    indices = [pos for pos, char in enumerate(results[0]) if char == "$"]
    if len(indices) <= 1:
        answer = results[0]
    else:
        answer = results[0][indices[0] + 1 : indices[-1]]
    # gold: 优先 doc["answer"]；若存在 solution 则从中抽 boxed
    gold = doc.get("answer")
    if gold is None and doc.get("solution"):
        gold = remove_boxed(last_boxed_only_string(doc["solution"]))
    if gold is not None and is_equiv(answer, gold):
        retval = 1
    return {"exact_match": retval}


# ----------------- 以下为 lm-eval 0.4.5 原样移植 -----------------
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    if s[: len(left)] != left:
        return s
    assert s[-1] == "}"
    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string
    except ValueError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # remove inverse spaces
    string = string.replace("\\!", "")
    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    # remove dollar signs
    string = string.replace("\\$", "")
    # remove units (on the right)
    string = remove_right_units(string)
    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)
    # remove spaces
    string = string.replace(" ", "")
    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)
    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"
    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)
    return string
