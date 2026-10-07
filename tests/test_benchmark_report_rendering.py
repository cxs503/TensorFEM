import zipfile
from tensorfem.benchmark_report_rendering import render_html, render_docx

def test_render_html_and_docx(tmp_path):
    report={"cases":{"demo":{"reference":{"value":1},"field_samples":[{"x":0,"y":0,"stress":[1,2]}],"relative_error":.01}},"error_policy":{"target_relative_error":.03}}
    h=render_html(report,tmp_path/"r.html"); assert h.exists() and "qualified" in h.read_text()
    d=render_docx(report,tmp_path/"r.docx"); assert zipfile.is_zipfile(d)
