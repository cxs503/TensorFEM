#!/usr/bin/env python3
"""Render one machine-readable benchmark report as HTML, PDF and/or DOCX."""
import argparse, json
from pathlib import Path
from tensorfem.benchmark_report_rendering import render_html, render_pdf, render_docx

p=argparse.ArgumentParser(); p.add_argument("report",type=Path); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--formats",default="html,pdf,docx")
a=p.parse_args(); data=json.loads(a.report.read_text()); a.output_dir.mkdir(parents=True,exist_ok=True); made=[]
for fmt in [x.strip().lower() for x in a.formats.split(',') if x.strip()]:
    fn=a.output_dir/(a.report.stem+({"html":".html","pdf":".pdf","docx":".docx"}[fmt]))
    {"html":render_html,"pdf":render_pdf,"docx":render_docx}[fmt](data,fn); made.append(str(fn))
print(json.dumps({"outputs":made,"schema":"tensorfem.benchmark-report-rendered/1.0"},indent=2))
