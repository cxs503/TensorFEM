"""Portable renderers for benchmark evidence (HTML, PDF and Word).

The renderer is deliberately dependency-light.  HTML is always available;
PDF uses matplotlib when installed and Word is a standards-compliant minimal
DOCX package (no python-docx dependency is required).  Rendering never turns
an over-3% result into a pass: blocked cases remain visibly marked.
"""
from __future__ import annotations
import html, json, math, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

SCHEMA = "tensorfem.benchmark-report-rendered/1.0"

def _cases(report):
    cases = report.get("cases", {})
    return cases.items() if isinstance(cases, dict) else ((c.get("name", "case"), c) for c in cases)

def _status(case):
    if case.get("status") in {"blocked", "failed"}: return "blocked"
    e = case.get("relative_error")
    return "qualified" if e is not None and float(e) <= .03 else "reference-only"

def _plot(case, title, path):
    """Create a deterministic field image when field samples are supplied."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    samples = case.get("field_samples", [])
    if not samples: return None
    xs=[float(s.get("x", i)) for i,s in enumerate(samples)]
    ys=[float(s.get("y", 0)) for s in samples]
    vals=[float((s.get("stress") or s.get("displacement") or [0])[0] if isinstance(s.get("stress") or s.get("displacement"), (list,tuple)) else (s.get("stress") or s.get("displacement") or 0)) for s in samples]
    fig, ax=plt.subplots(figsize=(6,4)); sc=ax.scatter(xs,ys,c=vals,cmap="turbo",s=90)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(sc, ax=ax, label="field value")
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig); return path

def render_html(report, output, title="TensorFEM Benchmark Report"):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); assets=output.parent/(output.stem+"_assets"); assets.mkdir(exist_ok=True)
    rows=[]; sections=[]
    for name,case in _cases(report):
        status=_status(case); err=case.get("relative_error"); image=_plot(case,name,assets/(name.replace("/","_")+".png"))
        rel=("../"+assets.name+"/"+image.name) if image else ""
        rows.append(f"<tr><td>{html.escape(name)}</td><td class='{status}'>{status}</td><td>{'' if err is None else f'{float(err)*100:.4f}%'}</td></tr>")
        sections.append(f"<section><h2>{html.escape(name)}</h2><p>Status: <b class='{status}'>{status}</b>. Reference and FE fields are shown as supplied; no result is promoted when error exceeds 3%.</p>"+(f"<img src='{html.escape(rel)}' alt='field cloud for {html.escape(name)}'>" if image else "<p>No field samples supplied.</p>")+f"<pre>{html.escape(json.dumps(case,indent=2,sort_keys=True))}</pre></section>")
    output.write_text("<!doctype html><meta charset='utf-8'><title>"+html.escape(title)+"</title><style>body{font:15px sans-serif;max-width:1100px;margin:2em auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:.5em}.qualified{color:green}.blocked{color:#b00;font-weight:bold}.reference-only{color:#a60}img{max-width:700px}pre{background:#f4f4f4;padding:1em;overflow:auto}</style><h1>"+html.escape(title)+"</h1><p>Schema: "+SCHEMA+"; acceptance target: 3% relative error.</p><table><tr><th>Case</th><th>Status</th><th>Relative error</th></tr>"+"".join(rows)+"</table>"+"".join(sections)+"\n",encoding="utf-8")
    return output

def render_pdf(report, output, title="TensorFEM Benchmark Report"):
    """Render a readable PDF; raises a clear error if matplotlib is absent."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc: raise RuntimeError("PDF rendering requires matplotlib") from exc
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(output) as pdf:
        fig=plt.figure(figsize=(8.27,11.69)); fig.text(.08,.94,title,fontsize=18,weight="bold"); fig.text(.08,.90,"Acceptance target: relative error <= 3%. Blocked results remain blocked.")
        y=.84
        for name,c in _cases(report):
            e=c.get("relative_error"); fig.text(.1,y,f"{name}: {_status(c)} | error: {'n/a' if e is None else f'{float(e)*100:.4f}%'}"); y-=.04
        pdf.savefig(fig); plt.close(fig)
        for name,c in _cases(report):
            fig=plt.figure(figsize=(8.27,11.69)); fig.text(.07,.95,name,fontsize=15,weight="bold"); fig.text(.07,.91,json.dumps(c,indent=2,sort_keys=True)[:7000],family="monospace",fontsize=7,va="top"); pdf.savefig(fig); plt.close(fig)
    return output

def render_docx(report, output, title="TensorFEM Benchmark Report"):
    """Write a minimal, portable DOCX without requiring python-docx."""
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    paras=[f"<w:p><w:r><w:t>{escape(title)}</w:t></w:r></w:p>", f"<w:p><w:r><w:t>Acceptance target: relative error &lt;= 3%; blocked results are not promoted.</w:t></w:r></w:p>"]
    for name,c in _cases(report):
        e=c.get("relative_error"); text=f"{name} — {_status(c)} — error: {'n/a' if e is None else f'{float(e)*100:.4f}%'}"
        paras.append(f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>")
        paras.append(f"<w:p><w:r><w:t>{escape(json.dumps(c,sort_keys=True)[:5000])}</w:t></w:r></w:p>")
    document='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(paras)+'<w:sectPr/></w:body></w:document>'
    ct='<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
    rel='<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",ct); z.writestr("_rels/.rels",rel); z.writestr("word/document.xml",document)
    return output
