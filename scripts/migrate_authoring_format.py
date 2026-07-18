"""Migrate repository post sources away from the legacy delimiter syntax."""

from __future__ import annotations

import argparse
import pathlib

from authoring_format import migrate_legacy_authoring


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite changed files")
    parser.add_argument("paths", nargs="*", default=["static/posts"])
    args = parser.parse_args()

    changed = []
    for raw_path in args.paths:
        path = pathlib.Path(raw_path)
        files = sorted(path.glob("*.md")) if path.is_dir() else [path]
        for markdown_file in files:
            source = markdown_file.read_text(encoding="utf-8")
            migrated = migrate_legacy_authoring(source)
            if ";;;" in migrated:
                raise RuntimeError("legacy delimiter remains in {}".format(markdown_file))
            if migrated == source:
                continue
            changed.append(markdown_file)
            if args.write:
                with markdown_file.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(migrated)

    action = "rewritten" if args.write else "would rewrite"
    for markdown_file in changed:
        print("{} {}".format(action, markdown_file))
    print("{} file(s) {}".format(len(changed), action))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
