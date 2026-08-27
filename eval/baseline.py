import argparse
import logging

def run_baseline(tasks_dir: str, model_id: str, output: str):
    logging.info(f"Running baseline for model {model_id} on tasks in {tasks_dir}")
    # Mock implementation of baseline runner
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    run_baseline(args.tasks, args.model, args.output)
