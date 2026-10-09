"""Rich exports must embed the actual figures and all result text."""
import zipfile

import pytest

from tensorfem.research_report_export import export_report


@pytest.fixture
def report_dependencies():
    for name in ('docx', 'reportlab', 'markdown', 'matplotlib'):
        pytest.importorskip(name)


def test_exports_embed_images_and_keep_result_text(tmp_path, report_dependencies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])
    fig.savefig(tmp_path/'field.png'); plt.close(fig)
    source = tmp_path/'case.md'
    (tmp_path/'results.json').write_text('{}')
    source.write_text('# 真实计算报告\n\n结果：应力误差 0.63553%。\n\n'
                      '| 网格 | 误差 |\n|---|---|\n| 80×16 | 0.63553% |\n\n'
                      '![真实场](field.png)\n\n未认证局部峰值。\n\n[原始结果](results.json)\n')
    h, p, d = export_report(source, tmp_path/'exports')
    assert 'data:image/png;base64,' in h.read_text()
    assert '0.63553%' in h.read_text() and '未认证局部峰值' in h.read_text()
    assert 'href="../results.json"' in h.read_text()
    with zipfile.ZipFile(d) as archive:
        xml = archive.read('word/document.xml').decode()
        assert '0.63553%' in xml and '未认证局部峰值' in xml
        assert any(name.startswith('word/media/') for name in archive.namelist())
    assert b'/Subtype /Image' in p.read_bytes()
    assert b'DejaVuSans' in p.read_bytes()  # numeric/Latin glyphs must render
    assert p.stat().st_size > 10000


def test_missing_figure_blocks_export(tmp_path, report_dependencies):
    source = tmp_path/'case.md'; source.write_text('# report\n\n![field](missing.png)')
    with pytest.raises(ValueError, match='missing report figure'):
        export_report(source, tmp_path/'exports')
    assert not list((tmp_path/'exports').glob('*.pdf'))
