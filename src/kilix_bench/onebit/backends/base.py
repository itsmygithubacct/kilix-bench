"""What every OneBit backend shares: the item rendering, the reply parsing,
the errors, and the usage numbers.

A backend is one module in this package with a class built on ``Backend``
and five module-level functions the command line and runner call:

    NAME                      the --backend value
    add_arguments(parser)     its own command-line options
    check(args, parser)       refuse bad option combinations
    spec(args)                its options as JSON-safe spec entries
    build(spec, env)          the Backend object for one run
    forbidden(spec)           local paths the result must not contain
    label(args)               default name for the result file

``ask(messages, arm)`` answers one item with a dict: ``reply`` (text or
None), ``latency_ms`` and, when the backend reports them, ``usage``
(``USAGE_KEYS``), log probabilities, exit status and so on. A backend
whose reply is not text (a tool call, a JSON object) parses it itself in
``parse``; everything else goes through ``contract.parse_reply``.
"""

from kilix_bench.onebit import contract

USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens",
              "reasoning_output_tokens")


class BackendError(Exception):
    """A call failed for a reason that is not the model's answer: network,
    server, usage limit, authentication. Never a format failure: the run
    stops, answers so far stay cached, and no finished result is written.

    ``kind`` names the cause when the backend can tell: limit, auth,
    network, timeout, or error.
    """

    def __init__(self, message, kind="error"):
        super().__init__(message)
        self.kind = kind


class ArmUnsupported(Exception):
    """The backend found it cannot run this arm (a rejected schema)."""


class Backend:
    kind = ""
    flavor = ""
    # Items asked at once; a network backend raises it.
    jobs = 1

    def cache_identity(self):
        """A string naming the model exactly, when answers may be cached and
        reused by a later run (a resumed run). None: never cached."""
        return None

    def default_arms(self):
        return ["raw", "constrained"]

    def render(self, item):
        """The chat every backend is given for one item."""
        return contract.messages(item)

    def parse(self, answer, arm):
        """0, 1 or None (a format failure) for one answer."""
        if "parsed" in answer:
            return answer["parsed"]
        return contract.parse_reply(answer.get("reply"))

    def count_tokens(self, messages):
        return None

    def describe(self):
        return {"flavor": self.flavor}

    def supports(self, arm):
        return False, f"unknown arm {arm}"

    def ask(self, messages, arm):
        raise NotImplementedError

    def settings(self):
        return {}

    def close(self):
        pass


def usage(input_tokens=None, cached_input_tokens=None, output_tokens=None,
          reasoning_output_tokens=None, **extra):
    """One answer's usage in the shared shape; missing numbers stay None."""
    out = {"input_tokens": input_tokens,
           "cached_input_tokens": cached_input_tokens,
           "output_tokens": output_tokens,
           "reasoning_output_tokens": reasoning_output_tokens}
    out.update({key: value for key, value in extra.items()
                if isinstance(value, int)})
    return out
