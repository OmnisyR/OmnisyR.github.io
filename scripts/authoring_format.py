"""Authoring helpers shared by the migration tool and the Gmeek patch."""

from __future__ import annotations

import html
import re


_DETAILS_RE = re.compile(r"(?is)<details\b(?P<attrs>[^>]*)>(?P<body>.*?)</details>")
_SUMMARY_RE = re.compile(r"(?is)<summary\b[^>]*>(?P<term>.*?)</summary>")
_LEGACY_NOTE_BLOCK_RE = re.compile(r"(?ms);;;a(.*?);;;a")
_LEGACY_NOTE_ENTRY_RE = re.compile(r"(?s);;;;(.*?);;;;")
_LEGACY_LANGUAGE_PAIR_RE = re.compile(r"(?s);;;e(.*?);;;e;;;c(.*?);;;c")


def _language_name(value: str) -> str:
    return "zh-CN" if value.lower() in {"zh", "zh-cn", "cn"} else "en"


def extract_note_definitions(text: str):
    """Remove semantic note blocks and return their Markdown definitions."""

    notes = []

    def collect(match):
        attrs = match.group("attrs")
        class_match = re.search(r"(?i)\bclass=[\"']([^\"']*)[\"']", attrs)
        classes = class_match.group(1).split() if class_match else []
        if "omnisyr-note" not in classes:
            return match.group(0)

        language_match = re.search(r"(?i)\blang=[\"']([^\"']+)[\"']", attrs)
        summary_match = _SUMMARY_RE.search(match.group("body"))
        if not language_match or not summary_match:
            return match.group(0)

        term = re.sub(r"<[^>]+>", "", summary_match.group("term"))
        note_body = _SUMMARY_RE.sub("", match.group("body"), count=1).strip()
        notes.append(
            {
                "language": _language_name(language_match.group(1)),
                "term": html.unescape(term).strip(),
                "body": note_body,
            }
        )
        return "\n"

    article = _DETAILS_RE.sub(collect, text or "")
    return article, notes


def _resolve_legacy_language(value: str, language: str) -> str:
    if language == "zh-CN":
        value = re.sub(r"(?s);;;e.*?;;;e", "", value)
        return re.sub(r"(?s);;;c(.*?);;;c", r"\1", value)
    value = re.sub(r"(?s);;;c.*?;;;c", "", value)
    return re.sub(r"(?s);;;e(.*?);;;e", r"\1", value)


def _note_details(language: str, term: str, body: str) -> str:
    return (
        '<details class="omnisyr-note" lang="{}">\n'
        "<summary>{}</summary>\n\n"
        "{}\n"
        "</details>"
    ).format(language, html.escape(term.strip()), body.strip())


def _migrate_legacy_notes(text: str) -> str:
    def replace_block(match):
        definitions = []
        for entry_match in _LEGACY_NOTE_ENTRY_RE.finditer(match.group(1)):
            entry = entry_match.group(1).strip()
            has_languages = ";;;e" in entry or ";;;c" in entry
            languages = ("en", "zh-CN") if has_languages else (
                "zh-CN" if re.search(r"[\u3400-\u9fff]", entry) else "en",
            )
            for language in languages:
                resolved = _resolve_legacy_language(entry, language).strip()
                if "::" not in resolved:
                    continue
                term, body = resolved.split("::", 1)
                if term.strip() and body.strip():
                    definitions.append(_note_details(language, term, body))
        if not definitions:
            return "\n"
        return "\n\n" + "\n\n".join(definitions) + "\n\n"

    return _LEGACY_NOTE_BLOCK_RE.sub(replace_block, text or "")


def migrate_legacy_authoring(text: str) -> str:
    """Convert old delimiter-based notes and language pairs to semantic HTML."""

    def replace_pair(match):
        english, chinese = match.group(1), match.group(2)
        is_block = (
            "\n" in english
            or "\n" in chinese
            or "Gmeek-html" in english
            or "Gmeek-html" in chinese
        )
        if is_block:
            return (
                '<div lang="en">\n\n{}\n\n</div>\n\n'
                '<div lang="zh-CN">\n\n{}\n\n</div>'
            ).format(english.strip(), chinese.strip())
        return (
            '<span lang="en">{}</span><span lang="zh-CN">{}</span>'
        ).format(english, chinese)

    migrated = _migrate_legacy_notes(text)
    return _LEGACY_LANGUAGE_PAIR_RE.sub(replace_pair, migrated)


def render_note_data(notes, markdown_renderer) -> str:
    """Render extracted note definitions into hidden, structured HTML."""

    if not notes:
        return ""

    articles = []
    for note in notes:
        articles.append(
            '<article lang="{}"><h6>{}</h6><div>{}</div></article>'.format(
                _language_name(note["language"]),
                html.escape(note["term"]),
                markdown_renderer(note["body"]),
            )
        )
    return (
        '<div class="omnisyr-denote-data" hidden aria-hidden="true">'
        + "".join(articles)
        + "</div>"
    )
