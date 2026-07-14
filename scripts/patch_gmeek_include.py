"""Patch Gmeek so issue bodies can include Markdown files from the repo.

The patch keeps the upstream generator intact except for `addOnePostJson`:
it resolves a small include directive before Gmeek computes descriptions,
custom post config, backups, and rendered pages.
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

        protected_denotes = []

        def protect_denotes(match):
            token = '@@OMNISYR_DENOTES_{}@@'.format(len(protected_denotes))
            protected_denotes.append(
                '<div class="omnisyr-denote-data" hidden aria-hidden="true">\n\n{}\n\n</div>'.format(match.group(0))
            )
            return token

        def replace_fence(match):
            lang = lang_name(match.group(1))
            body = match.group(2).strip()
            return '<div lang="{}">\n\n{}\n\n</div>'.format(lang, body)

        def replace_legacy_pair(match):
            english = match.group(1)
            chinese = match.group(2)
            is_block = '\n' in english or '\n' in chinese or 'Gmeek-html' in english or 'Gmeek-html' in chinese
            if is_block:
                return (
                    '<div lang="en">\n\n{}\n\n</div>\n\n'
                    '<div lang="zh-CN">\n\n{}\n\n</div>'
                ).format(english.strip(), chinese.strip())
            return '<span lang="en">{}</span><span lang="zh-CN">{}</span>'.format(english, chinese)

        text = re.sub(r'(?ms);;;a.*?;;;a', protect_denotes, text)
        text = re.sub(r'(?ms)^:::\s*(en|zh|zh-CN|cn)\s*$\n(.*?)^:::\s*$', replace_fence, text)
        text = re.sub(r'(?ms)^<!--\s*lang:\s*(en|zh|zh-CN|cn)\s*-->\s*$\n(.*?)^<!--\s*/lang\s*-->\s*$', replace_fence, text)
        text = re.sub(r'(?s);;;e(.*?);;;e;;;c(.*?);;;c', replace_legacy_pair, text)

        for index, denote in enumerate(protected_denotes):
            text = text.replace('@@OMNISYR_DENOTES_{}@@'.format(index), denote)
        return text

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
    source = source.replace("    def addOnePostJson(self,issue):\n", HELPER + "\n    def addOnePostJson(self,issue):\n", 1)
    source = source.replace("        if len(issue.labels)>=1:\n", "        if len(issue.labels)>=1:\n            issue_body=self.resolveIssueBody(issue)\n", 1)
    source = source.replace("            if issue.body==None:\n", "            if issue_body==None:\n", 1)
    source = source.replace("                self.blogBase[listJsonName][postNum][\"wordCount\"]=len(issue.body)\n", "                self.blogBase[listJsonName][postNum][\"wordCount\"]=len(issue_body)\n", 1)
    source = source.replace("                self.blogBase[listJsonName][postNum][\"description\"]=issue.body.split(period)[0].replace(\"\\\"\", \"\\'\")+period\n", "                self.blogBase[listJsonName][postNum][\"description\"]=issue_body.split(period)[0].replace(\"\\\"\", \"\\'\")+period\n", 1)
    source = source.replace("                postConfig=json.loads(issue.body.split(\"\\r\\n\")[-1:][0].split(\"##\")[1])\n", "                postConfig=json.loads(re.findall(r'##({[\\s\\S]*?})##', issue_body)[-1])\n", 1)
    source = source.replace("            if issue.body==None:\n                f.write('')\n            else:\n                f.write(issue.body)\n", "            if issue_body==None:\n                f.write('')\n            else:\n                f.write(issue_body)\n", 1)

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
