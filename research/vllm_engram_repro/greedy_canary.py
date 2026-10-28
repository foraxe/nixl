"""Record deterministic P/D output hashes without retaining prompt or output text."""

import argparse
import hashlib
import json
from pathlib import Path

import httpx

PROMPTS = [
    "The capital of France is",
    "A short explanation of gravity:",
    "The next integer after forty-one is",
    "Summarize this passage: " + "Rain falls on the old bridge. " * 20,
    "Repeat the next word once: river",
    "Translate to English: bonjour",
    "A function that adds two integers in Python:",
    "Consider this story: " + "The traveller carried a lantern. " * 80,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-salt-prefix")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    records = []
    with httpx.Client(timeout=180, trust_env=False) as client:
        for i, prompt in enumerate(PROMPTS):
            payload = {"model": args.model, "prompt": prompt, "temperature": 0,
                       "max_tokens": 16, "ignore_eos": True}
            if args.cache_salt_prefix is not None:
                payload["cache_salt"] = f"{args.cache_salt_prefix}-{i}"
            response = client.post(
                args.url.rstrip("/") + "/v1/completions",
                json=payload,
            )
            response.raise_for_status()
            value = response.json()
            text = value["choices"][0]["text"]
            usage = value["usage"]
            if usage["completion_tokens"] != 16:
                raise RuntimeError("unexpected completion token count")
            records.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "prompt_tokens": usage["prompt_tokens"],
                            "completion_tokens": usage["completion_tokens"]})
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    print(f"PASS {len(records)} greedy P/D canaries")


if __name__ == "__main__":
    main()
