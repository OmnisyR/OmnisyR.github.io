"""Patch Gmeek for repository posts and the blog's authoring format.

The patch resolves include directives, extracts semantic note definitions,
builds clean descriptions, and leaves math loading to the blog asset pipeline.
"""

from __future__ import annotations

import pathlib
import re
import sys


HELPER = r'''
    def normalizeLanguageBlocks(self, text):
        def lang_name(raw):
            raw = raw.lower()
            if raw in ['zh', 'zh-cn', 'cn']:
                return 'zh-CN'
            return 'en'

        def replace_fence(match):
            lang = lang_name(match.group(1))
            body = match.group(2).strip()
            return '<div lang="{}">\n\n{}\n\n</div>'.format(lang, body)

        text = migrate_legacy_authoring(text or '')
        text = re.sub(r'(?ms)^:::\s*(en|zh|zh-CN|cn)\s*$\n(.*?)^:::\s*$', replace_fence, text)
        text = re.sub(r'(?ms)^<!--\s*lang:\s*(en|zh|zh-CN|cn)\s*-->\s*$\n(.*?)^<!--\s*/lang\s*-->\s*$', replace_fence, text)
        return text

    def splitDenoteData(self, text):
        return extract_note_definitions(text or '')

    def issueDescription(self, text, title=''):
        source, unused_notes = extract_note_definitions(text or '')
        source = re.sub(
            r'(?is)<div\s+class=["\']omnisyr-denote-data["\'][^>]*>.*?</div>',
            ' ',
            source,
        )
        source = re.sub(r'(?ms);;;a.*?;;;a', ' ', source)
        source = re.sub(r'(?is)<!--.*?-->', ' ', source)
        source = re.sub(r'(?ms)```.*?```', ' ', source)
        source = re.sub(r'(?is)`Gmeek-html.*?`', ' ', source)

        def plain_text(markdown):
            value = markdown or ''
            value = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', value)
            value = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', value)
            value = re.sub(r'`([^`]*)`', r'\1', value)
            value = re.sub(r'\$[^$]*\$', ' ', value)
            value = re.sub(r'https?://\S+', ' ', value)
            value = re.sub(r'[*_~|]', '', value)

            paragraph = []
            for raw_line in value.splitlines():
                if re.search(r'<span\s+lang=', raw_line, re.I):
                    if paragraph:
                        break
                    continue
                line = re.sub(r'<[^>]+>', ' ', raw_line)
                line = re.sub(r'\s+', ' ', line).strip()
                if not line:
                    if paragraph:
                        break
                    continue
                if re.match(r'^(?:#{1,6}\s|>|[-+*]\s|\d+[.)]\s|\$\$|\\\[|;;;)', line):
                    if paragraph:
                        break
                    continue
                if len(line) < 20 and not paragraph:
                    continue
                paragraph.append(line)
                if len(' '.join(paragraph)) >= 180:
                    break

            result = re.sub(r'\s+', ' ', ' '.join(paragraph)).strip()
            if len(result) <= 160:
                return result
            shortened = result[:157]
            if ' ' in shortened:
                shortened = shortened.rsplit(' ', 1)[0]
            return shortened.rstrip(' ,.;:') + '...'

        descriptions = {}
        block_pattern = r'(?is)<div\s+lang=["\'](en|zh-CN)["\']>\s*(.*?)\s*</div>'
        for language, body in re.findall(block_pattern, source):
            description = plain_text(body)
            if description and language not in descriptions:
                descriptions[language] = description

        if descriptions.get('en') and descriptions.get('zh-CN'):
            return descriptions['en'] + ' || ' + descriptions['zh-CN']
        if descriptions:
            return descriptions.get('en') or descriptions.get('zh-CN')

        fallback = plain_text(source)
        if fallback:
            return fallback

        title_parts = [part.strip() for part in str(title or '').split('||') if part.strip()]
        return ' || '.join(title_parts[:2])

    def resolveIssueBody(self, issue):
        body = issue.body or ''
        match = re.search(r'<!--\s*(?:gmeek:include|include)\s+([^>]+?)\s*-->', body, re.I)
        if not match:
            return self.normalizeLanguageBlocks(body)

        include_path = match.group(1).strip().strip('\'"')
        include_path = include_path.replace('\\', '/')
        root = pathlib.Path.cwd().resolve()
        target = (root / include_path).resolve()

        if root not in target.parents and target != root:
            raise Exception('include path escapes repository: {}'.format(include_path))
        if target.suffix.lower() not in ['.md', '.markdown', '.txt']:
            raise Exception('include path must be Markdown or text: {}'.format(include_path))
        if not target.exists():
            raise Exception('include path not found: {}'.format(include_path))

        included = target.read_text(encoding='utf-8')

        config_match = re.search(r'<!--\s*##({[\s\S]*?})##\s*-->\s*$', body.strip())
        included_lines = included.splitlines()
        included_last_line = included_lines[-1] if included_lines else ''
        if config_match and '##' not in included_last_line:
            included = included.rstrip() + '\n\n<!-- ##{}## -->\n'.format(config_match.group(1))

        return self.normalizeLanguageBlocks(included)
'''


def ensure_import(source: str, import_line: str) -> str:
    if import_line in source:
        return source

    lines = source.splitlines(keepends=True)
    insert_at = 0
    seen_import = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("import ") or line.startswith("from "):
            seen_import = True
            insert_at = index + 1
            continue
        if seen_import:
            break

    lines.insert(insert_at, import_line)
    return "".join(lines)


def patch_gmeek(path: pathlib.Path) -> None:
    source = path.read_text(encoding="utf-8")
    if "def resolveIssueBody(self, issue):" in source:
        print("Gmeek include patch already present")
        return

    source = ensure_import(source, "import pathlib\n")
    source = ensure_import(
        source,
        "from scripts.authoring_format import extract_note_definitions, migrate_legacy_authoring, render_note_data\n",
    )
    source = source.replace("    def addOnePostJson(self,issue):\n", HELPER + "\n    def addOnePostJson(self,issue):\n", 1)
    source = source.replace("        if len(issue.labels)>=1:\n", "        if len(issue.labels)>=1:\n            issue_body=self.resolveIssueBody(issue)\n", 1)
    source = source.replace("            if issue.body==None:\n", "            if issue_body==None:\n", 1)
    source = source.replace("                self.blogBase[listJsonName][postNum][\"wordCount\"]=len(issue.body)\n", "                self.blogBase[listJsonName][postNum][\"wordCount\"]=len(issue_body)\n", 1)
    source = source.replace("                self.blogBase[listJsonName][postNum][\"description\"]=issue.body.split(period)[0].replace(\"\\\"\", \"\\'\")+period\n", "                self.blogBase[listJsonName][postNum][\"description\"]=self.issueDescription(issue_body, issue.title).replace(\"\\\"\", \"\\'\")\n", 1)
    source = source.replace("                postConfig=json.loads(issue.body.split(\"\\r\\n\")[-1:][0].split(\"##\")[1])\n", "                postConfig=json.loads(re.findall(r'##({[\\s\\S]*?})##', issue_body)[-1])\n", 1)
    source = source.replace("            if issue.body==None:\n                f.write('')\n            else:\n                f.write(issue.body)\n", "            if issue_body==None:\n                f.write('')\n            else:\n                f.write(issue_body)\n", 1)
    source = source.replace(
        "        post_body=self.markdown2html(f.read())\n        f.close()\n",
        "        post_source, note_definitions=self.splitDenoteData(f.read())\n"
        "        post_body=self.markdown2html(post_source)\n"
        "        if note_definitions:\n"
        "            post_body=render_note_data(note_definitions, self.markdown2html)+post_body\n"
        "        f.close()\n",
        1,
    )
    source = source.replace(
        "            issue[\"script\"]=issue[\"script\"]+'<script>MathJax = {tex: {inlineMath: [[\"$\", \"$\"]]}};</script><script async src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>'\n",
        "",
        1,
    )

    path.write_text(source, encoding="utf-8")
    print("Patched Gmeek include support in {}".format(path))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_gmeek_include.py /path/to/Gmeek.py", file=sys.stderr)
        return 2

    patch_gmeek(pathlib.Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
