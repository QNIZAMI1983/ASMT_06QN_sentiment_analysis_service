
#!/usr/bin/env python3
"""Command-line client for the sentiment analysis service."""

import argparse
import json
import os
import sys
import requests

DEFAULT_URL = os.environ.get("SERVICE_URL", "http://localhost:8080")


def build_parser():
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI Tool")
    parser.add_argument("--text", action="append", help="Text to analyse. Repeatable.")
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Base URL (default: {DEFAULT_URL})")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON response.")
    return parser


def collect_texts(args, parser):
    texts = list(args.text or [])
    if args.stdin:
        piped = sys.stdin.read().strip()
        if piped:
            texts.append(piped)
    if not texts:
        parser.error("provide at least one --text, or use --stdin")
    return texts


def format_result(result):
    sentiment = result["sentiment"].upper()
    snippet = result["text"]
    if len(snippet) > 60:
        snippet = snippet[:57] + "..."
    return f"{sentiment:<9} ({result['confidence']:.2%})  {snippet}"


def main():
    parser = build_parser()
    args = parser.parse_args()
    texts = collect_texts(args, parser)

    endpoint = f"{args.url.rstrip('/')}/predict"
    payload = {"text": texts if len(texts) > 1 else texts[0]}

    try:
        response = requests.post(endpoint, json=payload, timeout=60)
    except requests.exceptions.RequestException as exc:
        print(f"Could not reach {endpoint}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        body = response.json()
    except ValueError:
        print(f"Non-JSON response ({response.status_code}): {response.text}", file=sys.stderr)
        sys.exit(1)

    if args.raw or body.get("status") != "success":
        print(json.dumps(body, indent=4))
        sys.exit(0 if response.ok else 1)

    prediction = body["prediction"]
    results = prediction if isinstance(prediction, list) else [prediction]
    for result in results:
        print(format_result(result))


if __name__ == "__main__":
    main()