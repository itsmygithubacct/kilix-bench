"""Run upstream quality and throughput benchmarks with their native scoring."""

import argparse
import subprocess

SUITES = ("boolq", "ifeval", "gsm8k", "llama-bench")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def run(suite, argv):
    parser = argparse.ArgumentParser(prog=f"kilix-bench --suite {suite}")
    if suite == "llama-bench":
        parser.add_argument("--model-file", required=True, help="local GGUF file")
        parser.add_argument("--prompt-tokens", type=positive, default=512)
        parser.add_argument("--generate-tokens", type=positive, default=128)
        parser.add_argument("--repeats", type=positive, default=5)
        args = parser.parse_args(argv)
        command = ["llama-bench", "-m", args.model_file, "-p", str(args.prompt_tokens),
                   "-n", str(args.generate_tokens), "-r", str(args.repeats), "-o", "json"]
    else:
        parser.add_argument("--backend", required=True, help="lm-eval model backend")
        parser.add_argument("--model-args", required=True, help="lm-eval backend arguments")
        parser.add_argument("--limit", type=positive, help="run a subset for a smoke test")
        parser.add_argument("--output", help="upstream results directory")
        parser.add_argument("--apply-chat-template", action="store_true")
        args = parser.parse_args(argv)
        command = ["lm-eval", "run", "--tasks", suite, "--model", args.backend,
                   "--model_args", args.model_args]
        if args.apply_chat_template:
            command.append("--apply_chat_template")
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        if args.output:
            command.extend(["--output_path", args.output])
    try:
        result = subprocess.run(command, check=False)
    except FileNotFoundError:
        parser.exit(127, f"{parser.prog}: install {command[0]} and add it to PATH\n")
    return result.returncode if result.returncode >= 0 else 128 - result.returncode
