from pprint import pprint

from . import (gsm8keval, humaneval, mbpp, c4news)

TASK_REGISTRY = {
    "humaneval": humaneval.HumanEval,
    "mbpp": mbpp.MBPP,
    "gsm8k": gsm8keval.GSM8k,
    "c4": c4news.C4News
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        task = TASK_REGISTRY[task_name]()
        return task
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
