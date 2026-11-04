"""Run the evaluation on a specific question file (e.g. the tuning half or the test half).

Run:  python -m src.evaluation.run_eval_file data/eval/tune.jsonl --tag tune
Everything after the file name is passed to run_eval (--tag, --window, --verifier-nli, --limit).
"""
import sys

import src.evaluation.run_eval as run_eval


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        raise SystemExit("usage: python -m src.evaluation.run_eval_file PATH [run_eval options]")
    eval_file = sys.argv.pop(1)
    original = run_eval.load_config

    def with_eval_file(*args, **kwargs):
        cfg = original(*args, **kwargs)
        cfg["evaluation"]["eval_file"] = eval_file
        return cfg

    run_eval.load_config = with_eval_file
    run_eval.main()


if __name__ == "__main__":
    main()
