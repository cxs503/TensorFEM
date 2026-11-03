import re, pathlib, collections
src = pathlib.Path("src/tensorfem"); tests = list(pathlib.Path("tests").rglob("*.py")) + list(pathlib.Path("scripts").rglob("*.py"))
blob = "\n".join(p.read_text(errors="ignore") for p in tests)
mods = sorted(p.stem for p in src.glob("*.py") if p.stem != "__init__")
zero = []
for m in mods:
    pats = [f"tensorfem.{m}", f"from .{m}", f"import {m}"]
    hits = sum(blob.count(p) for p in pats)
    if hits == 0:
        lines = [l for l in (src/f"{m}.py").read_text(errors="ignore").splitlines() if re.match(r"^(def |class )", l)]
        zero.append((m, len(lines), lines[:6]))
print("modules:", len(mods), "zero-ref:", len(zero))
for m, n, l in zero:
    print(f"-- {m} ({n} top-level syms)")
    for x in l: print("    ", x[:80])
