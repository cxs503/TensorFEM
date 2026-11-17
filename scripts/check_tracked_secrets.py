"""Reject credential files and recognizable tokens in tracked files; never print values.

This is a current-tree guard, not a historical scan or credential revocation.
Only paths and rule names are emitted, including on malformed input.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
from pathlib import Path
import re
import subprocess
import warnings

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERNS = {
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
JWT = re.compile(rb"\b(eyJ[A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]{16,})\b")


def _jwt_present(raw: bytes) -> bool:
    for match in JWT.finditer(raw):
        try:
            header, payload = (
                json.loads(base64.urlsafe_b64decode(part + b"=" * (-len(part) % 4)))
                for part in match.groups()[:2]
            )
            if isinstance(header, dict) and "alg" in header and isinstance(payload, dict):
                return True
        except (ValueError, UnicodeError):
            continue
    return False


def _credential_literal(raw: bytes) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value in {"pass", "password", "passwd", "pwd", "secret"}
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value
                ):
                    return True
    return False


def scan_paths(root: Path, paths: list[str]) -> list[dict[str, str]]:
    findings = []
    for name in paths:
        path = root / name
        if not path.is_file():
            continue  # tracked deletion in a working tree
        if path.name in {".tok", ".env"}:
            findings.append({"path": name, "rule": "credential_file"})
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        for rule, pattern in TOKEN_PATTERNS.items():
            if pattern.search(raw):
                findings.append({"path": name, "rule": rule})
        if _jwt_present(raw):
            findings.append({"path": name, "rule": "jwt"})
        if path.suffix == ".py" and _credential_literal(raw):
            findings.append({"path": name, "rule": "credential_literal"})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=args.root).decode().split("\0")
    findings = scan_paths(args.root, [name for name in names if name])
    print(
        json.dumps(
            {"passed": not findings, "scope": "tracked current tree", "findings": findings},
            indent=2,
        )
    )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
