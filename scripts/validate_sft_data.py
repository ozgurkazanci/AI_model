import argparse
import json
import logging
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from asic_ai.data.format import validate_sft_format

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Validate generated SFT data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--report", type=str, required=True, help="Output report JSON file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    report_path = Path(args.report)
    
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist.")
        sys.exit(1)
        
    valid_count = 0
    invalid_count = 0
    errors_by_line = {}
    
    with open(input_path, "r") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                is_valid, errors = validate_sft_format(messages)
                
                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    errors_by_line[idx + 1] = errors
            except json.JSONDecodeError:
                invalid_count += 1
                errors_by_line[idx + 1] = ["Invalid JSON format"]
                
    report = {
        "total_processed": valid_count + invalid_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "errors_by_line": errors_by_line
    }
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Validation complete. Valid: {valid_count}, Invalid: {invalid_count}")
    logger.info(f"Report saved to {report_path}")
    
    if invalid_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
