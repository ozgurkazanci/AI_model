#!/usr/bin/env python3
"""Extend the base tokenizer with circuit design domain tokens.

Usage:
    PYTHONPATH=src python scripts/extend_tokenizer.py --model Qwen/Qwen3.6-35B-A3B --output tokenizer_extended/
    PYTHONPATH=src python scripts/extend_tokenizer.py --model Qwen/Qwen3.6-35B-A3B --test-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asic_ai.tokenizer.extend import (
    DEFAULT_TEST_STRINGS,
    TokenExtensionConfig,
    extend_tokenizer,
    get_new_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extend_tokenizer")


def main():
    parser = argparse.ArgumentParser(description="Extend tokenizer with circuit design tokens")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="Base model/tokenizer name")
    parser.add_argument("--output", default="tokenizer_extended", help="Output directory")
    parser.add_argument("--test-only", action="store_true", help="Only show token list, don't extend")
    args = parser.parse_args()

    config = TokenExtensionConfig()
    tokens = get_new_tokens(config)

    if args.test_only:
        print(f"\nTokens to add ({len(tokens)}):")
        for i, t in enumerate(tokens):
            print(f"  {i+1:3d}. '{t}'")
        print(f"\nTest strings:")
        for s in DEFAULT_TEST_STRINGS:
            print(f"  - {s}")
        return

    # Extend tokenizer
    try:
        stats = extend_tokenizer(
            tokenizer_name_or_path=args.model,
            output_path=args.output,
            config=config,
            test_strings=DEFAULT_TEST_STRINGS,
        )

        print(f"\nTokenizer Extension Results:")
        print(f"  Original vocab: {stats['original_vocab_size']:,}")
        print(f"  New vocab:      {stats['new_vocab_size']:,}")
        print(f"  Tokens added:   {stats['tokens_added']}")
        print(f"  Already existed: {stats['tokens_already_existed']}")
        print(f"  Saved to:       {stats['output_path']}")

        if "tokenization_comparison" in stats:
            print(f"\nTokenization comparison:")
            for text, comp in stats["tokenization_comparison"].items():
                improvement = comp["improvement"]
                indicator = f"(-{improvement} tokens)" if improvement > 0 else "(no change)"
                print(f"  '{text}'")
                print(f"    Before: {comp['before_count']} tokens -> After: {comp['after_count']} tokens {indicator}")

    except ImportError:
        log.error("transformers package required. Install with: pip install transformers")
        sys.exit(1)
    except Exception as e:
        log.error(f"Failed to extend tokenizer: {e}")
        log.info("This is expected if the model is not downloaded. Use --test-only to preview tokens.")
        sys.exit(1)


if __name__ == "__main__":
    main()
