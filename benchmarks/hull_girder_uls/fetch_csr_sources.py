"""Re-fetch the anchor documents of the W24 CSR ULS study and verify their sha256.

The three documents behind ``csr_uls_anchor.json`` live in the public IACS
object store.  ``www.iacs.org.uk`` answers HTTP 403 through the execution
sandbox proxy (which is why missions 14512 and 15094 recorded BLOCKED /
NOT_OBTAINED), but ``iacs.s3.af-south-1.amazonaws.com`` answers anonymous
``GetObject`` requests, so the transcription is reproducible without any
credential.

Usage::

    python benchmarks/hull_girder_uls/fetch_csr_sources.py --dest /tmp/iacs2
    python benchmarks/hull_girder_uls/fetch_csr_sources.py --dest /tmp/iacs2 --list

``--list`` only performs the bucket listing (no download) and prints how many
objects are visible and where the documents sit.  Downloads are skipped when a
file of the right size and sha256 is already present, so the script is safe to
re-run.  The PDFs are *not* committed: only the URL, byte count and sha256 in
``csr_uls_anchor.json`` are, so the anchor stays small while remaining checkable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://iacs.s3.af-south-1.amazonaws.com"
USER_AGENT = "curl/8"

#: kept in sync with SOURCE_DOCUMENTS in src/tensorfem/csr_uls_factors.py
DOCUMENTS = [
    {
        "id": "AN-IACS-CSR",
        "key": "wp-content/uploads/2022/03/15144815/csr-h-01-jan-2019.pdf",
        "sha256": "ffb51b3153e43a7d3a2e840971bc948128152e1c9ffe66c3891d6e620e250e7f",
        "bytes": 21439977,
    },
    {
        "id": "AN-IACS-UR-S7",
        "key": "wp-content/uploads/2022/05/18114816/ur-s7rev4.pdf",
        "sha256": "25a82bfec5f2beeafb780cbf2a1498efbe1a1573f05b923b1cf9b9729cac2096",
        "bytes": 18903,
    },
    {
        "id": "AN-IACS-UR-S11",
        "key": "wp-content/uploads/2022/05/18115332/ur-s11rev10.pdf",
        "sha256": "4711024b4b537cd7e0c6203e01b5936575fc3f542349ec1addf589397986ed53",
        "bytes": 1501639,
    },
]


def _open(path: str) -> bytes:
    request = urllib.request.Request(path, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def list_documents(prefix: str = "wp-content/uploads") -> dict:
    """One listing page: how many keys the prefix exposes and a few examples."""
    url = f"{BUCKET}/?list-type=2&max-keys=20&prefix={urllib.parse.quote(prefix)}"
    body = _open(url).decode()
    keys = []
    cursor = 0
    while True:
        start = body.find("<Key>", cursor)
        if start < 0:
            break
        end = body.find("</Key>", start)
        keys.append(body[start + 5:end])
        cursor = end
    return {"prefix": prefix, "http": "200", "keys_in_page": len(keys),
            "examples": keys[:5]}


def fetch(dest: Path, *, verify_only: bool = False) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for document in DOCUMENTS:
        target = dest / Path(document["key"]).name
        status = "present"
        if not (target.exists() and target.stat().st_size == document["bytes"]):
            if verify_only:
                results.append({"id": document["id"], "path": str(target),
                                "status": "missing"})
                continue
            target.write_bytes(_open(f"{BUCKET}/{urllib.parse.quote(document['key'])}"))
            status = "downloaded"
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        results.append({
            "id": document["id"],
            "path": str(target),
            "status": status,
            "bytes": target.stat().st_size,
            "sha256": digest,
            "sha256_matches": digest == document["sha256"],
        })
    return {"documents": results,
            "all_verified": all(r.get("sha256_matches", False) for r in results)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default="/tmp/iacs2",
                        help="download directory (default /tmp/iacs2)")
    parser.add_argument("--list", action="store_true",
                        help="only prove the bucket is listable, do not download")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify local copies without downloading")
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps(list_documents(), indent=2))
        return 0
    report = fetch(Path(args.dest), verify_only=args.verify_only)
    print(json.dumps(report, indent=2))
    return 0 if report["all_verified"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())