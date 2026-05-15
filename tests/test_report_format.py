from src.core.report_format import strip_report_header


def test_strip_report_header_removes_title_and_date():
    report = """# BÁO CÁO CHẨN ĐOÁN BỆNH LÚA
Ngày báo cáo: 14/05/2026

## KẾT LUẬN CHẨN ĐOÁN
Rice Blast / Đạo ôn
"""

    cleaned = strip_report_header(report)

    assert "BÁO CÁO CHẨN ĐOÁN BỆNH LÚA" not in cleaned
    assert "Ngày báo cáo" not in cleaned
    assert cleaned.startswith("## KẾT LUẬN CHẨN ĐOÁN")


def test_strip_report_header_keeps_regular_report():
    report = "## KẾT LUẬN CHẨN ĐOÁN\nBrown Spot / Đốm nâu"

    assert strip_report_header(report) == report
