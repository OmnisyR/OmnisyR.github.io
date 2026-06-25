# Blog Authoring Notes

This blog is still powered by Gmeek and GitHub Issues, but long posts can now
live in repository Markdown files instead of issue bodies.

## Recommended Issue Body

For long posts, keep the issue body short:

```md
<!-- gmeek:include static/posts/10.md -->
```

Put the full article in `static/posts/10.md`. The number should match the
GitHub issue number. This avoids GitHub issue character limits and makes the
article easier to edit locally.

Per-post Gmeek config can stay inside the included file:

```md
<!-- ##{"script":"<script src='/assets/HyperTOC.js'></script>"}## -->
```

## Bilingual Content

For full bilingual sections in included Markdown files, use language fences:

```md
::: en

## Introduction

English content.

:::

::: zh

## 引言

中文内容。

:::
```

The build step converts these fences into HTML language blocks before Gmeek
renders Markdown. If you want the GitHub issue preview itself to render the
blocks, you can write the HTML form directly:

```md
<div lang="en">

## Introduction

English content.

</div>

<div lang="zh-CN">

## 引言

中文内容。

</div>
```

For short titles, labels, or subtitles where HTML is not practical, use:

```text
English title || 中文标题
```

The old `;;;e...;;;e;;;c...;;;c` format is still supported for existing posts,
but new content should use the cleaner forms above.

## URL Mode

`config.json` now uses:

```json
"urlMode": "issue"
```

Posts will be generated as `/post/3.html`, `/post/4.html`, and so on. This keeps
URLs stable even when issue titles change.

## Migrating Existing Issues

After committing `static/posts/*.md`, set a GitHub token and run:

```powershell
$env:GITHUB_TOKEN = "your_token"
python scripts/migrate_issues_to_static.py --apply --clean-title
```

Without `--apply`, the script only prints what it would change.
