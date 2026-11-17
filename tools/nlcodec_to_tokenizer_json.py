#!/usr/bin/env python3
"""Convert an NLCodec BPE model to a tokenizerspp tokenizer.json file.

NLCodec applies greedy longest-prefix segmentation to text normalized as
``"▁".join(text.split()) + "▁"``. A WordPiece model with an empty continuation
prefix and unfused unknown tokens reproduces that segmentation while retaining
the original token IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


SPACE_TOKEN = "▁"
DUPLICATE_ALIAS_PREFIX = "\U000F0000nlcodec_duplicate_"
UNICODE_WHITESPACE_PATTERN = (
    r"[\x{0009}-\x{000D}\x{001C}-\x{0020}\x{0085}\x{00A0}\x{1680}"
    r"\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]+"
)


class ConversionError(ValueError):
    """Raised when an input file is not a supported NLCodec BPE model."""


@dataclass(frozen=True)
class TokenType:
    idx: int
    name: str
    level: int
    frequency: int
    children: tuple[int, ...]


def read_nlcodec_model(path: Path) -> tuple[dict[str, Any], list[TokenType]]:
    with path.open(encoding="utf-8") as source:
        header_line = source.readline()
        if not header_line.startswith("#{"):
            raise ConversionError("NLCodec model is missing its JSON metadata header")
        try:
            metadata = json.loads(header_line[1:])
        except json.JSONDecodeError as error:
            raise ConversionError(f"invalid NLCodec metadata: {error}") from error

        tokens: list[TokenType] = []
        for line_number, raw_line in enumerate(source, start=2):
            columns = raw_line.rstrip("\r\n").split("\t")
            if len(columns) < 4:
                raise ConversionError(
                    f"line {line_number}: expected at least four tab-separated columns"
                )
            try:
                idx = int(columns[0])
                level = int(columns[2])
                frequency = int(columns[3])
                children = tuple(int(value) for value in columns[4].split()) \
                    if len(columns) > 4 else ()
            except ValueError as error:
                raise ConversionError(f"line {line_number}: invalid integer field") from error

            if idx != len(tokens):
                raise ConversionError(
                    f"line {line_number}: expected token ID {len(tokens)}, found {idx}"
                )
            if any(child >= idx or child < 0 for child in children):
                raise ConversionError(
                    f"line {line_number}: child IDs must refer to earlier tokens"
                )
            tokens.append(TokenType(idx, columns[1], level, frequency, children))

    if metadata.get("scheme") not in (None, "bpe"):
        raise ConversionError(f"unsupported NLCodec scheme: {metadata['scheme']!r}")
    if metadata.get("scheme") is None and metadata.get("max_level") != 1:
        raise ConversionError(
            "legacy NLCodec model is not BPE (expected max_level equal to 1)"
        )
    if metadata.get("total") not in (None, len(tokens)):
        raise ConversionError(
            f"metadata declares {metadata['total']} tokens, but the file contains {len(tokens)}"
        )
    if not tokens:
        raise ConversionError("NLCodec model has no tokens")
    return metadata, tokens


def make_tokenizer(tokens: list[TokenType], unk_token: str) -> tuple[dict[str, Any], list[TokenType]]:
    source_names = {token.name for token in tokens}
    vocab: dict[str, int] = {}
    first_ids: dict[str, int] = {}
    duplicates: list[TokenType] = []
    duplicate_aliases: list[tuple[str, str]] = []
    for token in tokens:
        if token.name in first_ids:
            duplicates.append(token)
            alias = f"{DUPLICATE_ALIAS_PREFIX}{token.idx}"
            if alias in source_names or alias in vocab:
                raise ConversionError(f"duplicate-token alias collision: {alias!r}")
            vocab[alias] = token.idx
            duplicate_aliases.append((alias, token.name))
        else:
            first_ids[token.name] = token.idx
            vocab[token.name] = token.idx

    if unk_token not in first_ids:
        raise ConversionError(f"unknown token {unk_token!r} is absent from the vocabulary")
    if SPACE_TOKEN not in vocab:
        raise ConversionError(f"NLCodec space token {SPACE_TOKEN!r} is absent from the vocabulary")

    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Strip", "strip_left": True, "strip_right": True},
                {
                    "type": "Replace",
                    "pattern": {"Regex": UNICODE_WHITESPACE_PATTERN},
                    "content": SPACE_TOKEN,
                },
                {
                    "type": "Replace",
                    "pattern": {"Regex": "$"},
                    "content": SPACE_TOKEN,
                },
            ],
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": SPACE_TOKEN},
            "behavior": "MergedWithPrevious",
            "invert": False,
        },
        "post_processor": None,
        "decoder": {
            "type": "Sequence",
            "decoders": [
                *(
                    {
                        "type": "Replace",
                        "pattern": {"String": alias},
                        "content": token_name,
                    }
                    for alias, token_name in duplicate_aliases
                ),
                {"type": "BPE", "suffix": SPACE_TOKEN},
            ],
        },
        "model": {
            "type": "WordPiece",
            "unk_token": unk_token,
            "continuing_subword_prefix": "",
            "max_input_chars_per_word": 2_147_483_647,
            "fuse_unk": False,
            "vocab": vocab,
        },
    }
    return tokenizer, duplicates


def write_json(tokenizer: dict[str, Any], destination: Path | None, pretty: bool) -> None:
    dump_args = {
        "ensure_ascii": False,
        "indent": 2 if pretty else None,
        "separators": None if pretty else (",", ":"),
    }
    if destination is None:
        json.dump(tokenizer, sys.stdout, **dump_args)
        sys.stdout.write("\n")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(tokenizer, output, **dump_args)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.chmod(0o644)
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an NLCodec BPE model to tokenizerspp tokenizer.json format."
    )
    parser.add_argument("input", type=Path, help="NLCodec .model file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="output tokenizer.json; omit to write to standard output",
    )
    parser.add_argument("--unk-token", default="<unk>", help="unknown token string")
    parser.add_argument("--pretty", action="store_true", help="indent the output JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _, tokens = read_nlcodec_model(args.input)
        tokenizer, duplicates = make_tokenizer(tokens, args.unk_token)
        if args.output is not None and args.input.resolve() == args.output.resolve():
            raise ConversionError("input and output paths must differ")
        write_json(tokenizer, args.output, args.pretty)
    except (ConversionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    destination = str(args.output) if args.output is not None else "standard output"
    print(
        f"converted {len(tokens)} tokens to {destination}",
        file=sys.stderr,
    )
    if duplicates:
        print(
            f"note: represented {len(duplicates)} duplicate token spellings with internal aliases; "
            "encoding uses their first IDs, matching NLCodec",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())