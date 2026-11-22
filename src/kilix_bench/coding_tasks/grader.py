"""Run inside the disposable grading namespace, using the system Python."""

import ast
import json
import multiprocessing
import os
import resource
import sys


def check(task, doc, answer):
    if task == "output":
        return ast.literal_eval(answer.strip()) == ast.literal_eval(doc["output"])

    if task == "input":
        call = ast.parse(f"f({answer})", mode="eval").body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "f":
            return False
        namespace = {}
        exec(doc["code"], namespace)
        value = eval(compile(ast.Expression(call), "<prediction>", "eval"), namespace)
        return value == ast.literal_eval(doc["output"])

    from evalplus.eval import PASS, untrusted_check
    from evalplus.gen.util import trusted_exec

    reference = doc["prompt"] + doc["canonical_solution"]
    candidate = doc["prompt"] + answer
    for field in ("base_input", "plus_input"):
        inputs = json.loads(doc[field])
        expected, timings = trusted_exec(reference, inputs, doc["entry_point"], record_time=True)
        result, _ = untrusted_check(dataset="humaneval", entry_point=doc["entry_point"],
                                    code=candidate, inputs=inputs, expected=expected,
                                    atol=doc["atol"], ref_time=timings, fast_check=True)
        if result != PASS:
            return False
    return True


def isolate_candidate():
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)


if __name__ == "__main__":
    task, doc, answer = json.load(sys.stdin)
    if task == "humaneval":
        sys.path.insert(0, "/evalplus.whl")
        from evalplus.eval import untrusted_check
        from evalplus.gen.util import trusted_exec
        multiprocessing.set_start_method("fork")
        os.register_at_fork(after_in_child=isolate_candidate)
    else:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    if doc is None:  # Runtime/dependency check before generating any answers.
        sys.exit(0)
    for limit, value in ((resource.RLIMIT_CPU, 120 if task == "humaneval" else 20), (resource.RLIMIT_AS, 1024 ** 3),
                         (resource.RLIMIT_FSIZE, 1024 ** 2),
                         (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(limit, (value, value))
    if task == "humaneval":
        passed = check(task, doc, answer)
    else:
        isolate_candidate()
        try:
            passed = check(task, doc, answer)
        except BaseException:
            passed = False
    # A candidate that exits Python with status zero has not passed the tests.
    sys.exit(42 if passed else 0)
