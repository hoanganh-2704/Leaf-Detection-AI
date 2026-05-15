import re
import unicodedata


def _normalize_heading(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def strip_report_header(report: str) -> str:
    """Remove generated report title/date boilerplate from UI-facing text."""
    lines = report.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    if lines:
        first_line = _normalize_heading(lines[0].lstrip("#*- "))
        is_report_title = "bao cao" in first_line and (
            "chan doan" in first_line or "chuan doan" in first_line
        )
        if is_report_title:
            lines.pop(0)

            while lines:
                normalized = _normalize_heading(lines[0].lstrip("#*- "))
                if not normalized or normalized.startswith("ngay bao cao"):
                    lines.pop(0)
                    continue
                break

    return "\n".join(lines).strip()
