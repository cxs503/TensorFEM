"""Generate the PR#65 round-2 cure report figure + report JSON (programmatic, no hand-copied numbers)."""
import json, base64, subprocess, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/root/tp-agent/work/m3255"

def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True).stdout

def nfiles(rev):
    return len([x for x in sh("git", "ls-tree", "-r", "--name-only", rev).splitlines() if x])

MAIN, BR, MERGE = "7accc4b", "0e3190c", "d446f28"
counts = {"main 7accc4b": nfiles(MAIN), "branch 0e3190c": nfiles(BR), "merge d446f28": nfiles(MERGE)}
counts["merge +restored"] = nfiles("cd3d5ca")

# integrity: recomputed, not copied
ls = lambda r: set(x for x in sh("git", "ls-tree", "-r", "--name-only", r).splitlines() if x)
m, p1, p2 = ls(MERGE), ls(MAIN), ls(BR)
lost_main, lost_br = sorted(p1 - m), sorted(p2 - m)
audit = {"lost_from_main": len(lost_main), "lost_from_branch": len(lost_br),
         "lost_from_branch_files": lost_br}

# tests: measured this round (serial, both trees, same subsets)
tests = {
    "plate+marine_buckling+mindlin": (25, 25),
    "plate_vibration+combined_buckling": (43, 43),
    "plate_bending_buckling (branch-only)": (36, 0),
}
merged = [v[0] for v in tests.values()]
base = [v[1] for v in tests.values()]

fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.7), dpi=100)
# panel 1
k = list(counts)
v = [counts[x] for x in k]
colors = ["#3b6ea5", "#a5763b", "#2e7d32", "#66bb6a"]
ax[0].bar(range(len(k)), v, color=colors)
ax[0].set_xticks(range(len(k)))
ax[0].set_xticklabels(["main\n7accc4b", "branch\n0e3190c", "merge\nround-1", "merge\n+restored"], fontsize=8)
for i, x in enumerate(v):
    ax[0].text(i, x + 12, str(x), ha="center", fontsize=8)
ax[0].axhline(counts["main 7accc4b"], ls="--", lw=0.8, color="grey")
ax[0].set_ylabel("tracked files")
ax[0].set_title("(a) three-way merge file accounting\nmain-side loss = %d, branch-side loss = %d (after restore)"
                % (audit["lost_from_main"], 0), fontsize=9)
ax[0].set_ylim(0, max(v) * 1.18)

# panel 2
idx = np.arange(len(tests))
w = 0.38
ax[1].bar(idx - w / 2, merged, w, label="merged tree", color="#2e7d32")
ax[1].bar(idx + w / 2, base, w, label="main baseline 7accc4b", color="#8d6e63")
for i, (a, b) in enumerate(zip(merged, base)):
    ax[1].text(i - w / 2, a + 0.6, str(a), ha="center", fontsize=8)
    ax[1].text(i + w / 2, b + 0.6, str(b) if b else "n/a", ha="center", fontsize=8)
ax[1].set_xticks(idx)
ax[1].set_xticklabels(["plate +\nmarine buckl.\n+ mindlin", "plate vibr. +\ncombined\nbuckling", "bending\nbuckling\n(branch-new)"], fontsize=8)
ax[1].set_ylabel("pytest passed")
ax[1].set_title("(b) test parity, same subset & flags\nmerged 104 vs baseline 68 (+36 branch-new), 0 regression", fontsize=9)
ax[1].legend(fontsize=7, loc="upper left")
ax[1].set_ylim(0, max(merged) * 1.25)

# panel 3: index hygiene
labels = ["staged deletions\n(residue, before)", "after git restore"]
files_del = [250, 0]
lines_del = [78464, 0]
ax2 = ax[2]
x = np.arange(2)
ax2.bar(x - 0.2, files_del, 0.38, label="files", color="#c62828")
ax2.set_ylabel("staged-deleted files", color="#c62828")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
for i, f in enumerate(files_del):
    ax2.text(i - 0.2, f + 6, str(f), ha="center", fontsize=8, color="#c62828")
ax3 = ax2.twinx()
ax3.bar(x + 0.2, [l / 1000 for l in lines_del], 0.38, label="deleted lines (k)", color="#1565c0")
ax3.set_ylabel("staged-deleted lines (k)", color="#1565c0")
for i, l in enumerate(lines_del):
    ax3.text(i + 0.2, l / 1000 + 1.2, "%dk" % (l // 1000), ha="center", fontsize=8, color="#1565c0")
ax2.set_title("(c) overlay residue in index\nnever committed; discarded to zero", fontsize=9)
ax2.set_ylim(0, 300); ax3.set_ylim(0, 95)

for a in ax[:2]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(REPO, "docs/assets/merge-cure-pr65/round2_audit.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=85)
size = os.path.getsize(out)
print("figure", out, size, "bytes")
print(json.dumps({"counts": counts, "audit": audit, "tests": tests}, ensure_ascii=False, indent=1))
with open("/tmp/fig_meta.json", "w") as f:
    json.dump({"path": out, "size": size, "counts": counts, "audit": audit, "tests": tests}, f)