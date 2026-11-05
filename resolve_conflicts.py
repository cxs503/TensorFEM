#!/usr/bin/env python3
"""Resolve merge conflicts for PR #61 (agent/m3202 -> main)."""

import os

def resolve_changelog():
    with open('CHANGELOG.md', 'r') as f:
        content = f.read()

    head_start = content.find('<<<<<<< HEAD')
    sep = content.find('\n=======\n')
    tail_start = content.find('\n>>>>>>> origin-agent-m3202')

    head_content = content[head_start+len('<<<<<<< HEAD\n'):sep].strip()
    tail_content = content[sep+len('\n=======\n'):tail_start].strip()

    before = content[:head_start]
    after = content[tail_start+len('\n>>>>>>> origin-agent-m3202\n'):]

    resolved = before + head_content + '\n\n' + tail_content + '\n' + after

    with open('CHANGELOG.md', 'w') as f:
        f.write(resolved)
    print('CHANGELOG.md resolved')

def resolve_marine_docs():
    with open('docs/marine-hydrostatics.md', 'r') as f:
        content = f.read()

    head_start = content.find('<<<<<<< HEAD')
    sep = content.find('\n=======\n')
    tail_start = content.find('\n>>>>>>> origin-agent-m3202')

    # Semantic merge: acknowledge large-angle GZ is now available separately
    # while keeping the limitations
    resolved_section = (
        "Large-angle `GZ` curves and `KN` cross-curves are provided separately by\n"
        "`tensorfem.marine_large_angle_gz` (see `docs/marine-large-angle-stability.md`),\n"
        "which solves the exact clipped-section geometry for prismatic barges and is\n"
        "verified against the wall-sided closed form to machine precision.\n"
        "\n"
        "This module still is not a large-angle `GZ` curve, cross-curves, free-surface, damage"
    )

    before = content[:head_start]
    after = content[tail_start+len('\n>>>>>>> origin-agent-m3202\n'):]

    resolved = before + resolved_section + '\n' + after

    with open('docs/marine-hydrostatics.md', 'w') as f:
        f.write(resolved)
    print('docs/marine-hydrostatics.md resolved')

if __name__ == '__main__':
    os.chdir('/root/tp-agent/work/m3900')
    resolve_changelog()
    resolve_marine_docs()
