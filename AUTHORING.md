# Blog Authoring Guide

The blog is powered by Gmeek and GitHub Issues, while long articles live in
repository Markdown files so they are not limited by the issue body size.

## Issue Body

Keep a long-post issue body short:

```md
<!-- gmeek:include static/posts/10.md -->
```

Put the complete article in `static/posts/10.md`. The filename should match the
GitHub issue number.

## Bilingual Content

Use standard HTML language attributes for complete Markdown sections:

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

Use spans only for short inline text:

```md
## <span lang="en">Methods</span><span lang="zh-CN">方法</span>
```

Complete language blocks are preferred because they keep links, headings, and
shared words from appearing in both languages at once.

Issue titles and labels can use `English || 中文` because those values are not
part of the Markdown source.

## Annotations

Write each translated annotation as a semantic `details` block:

```md
<details class="omnisyr-note" lang="en">
<summary>Markov chain</summary>

The state at a given moment only depends on the previous state.
</details>

<details class="omnisyr-note" lang="zh-CN">
<summary>马尔可夫链</summary>

某一时刻的状态只与上一时刻的状态相关。
</details>
```

Reference the corresponding term with inline code in the article:

```md
The forward process is a `Markov chain`.
```

Annotation bodies support Markdown links, code, and LaTeX. The build removes
the definition blocks from the article and gives the structured data to
`HyperTOC.js`.

## Mathematics

Use normal TeX delimiters:

```md
Inline: $x_t$

Display:
$$
x_t = \sqrt{\alpha_t}x_0 + \sqrt{1 - \alpha_t}\epsilon
$$
```

`MathLoader.js` loads MathJax 4 only when an article or annotation contains
math. Do not add a MathJax script to individual posts.

## URLs

`config.json` uses `"urlMode": "issue"`, so article URLs remain stable as
`/post/3.html`, `/post/4.html`, and so on even when issue titles change.

## Migrating Issues

After committing `static/posts/*.md`, set a GitHub token and run:

```powershell
$env:GITHUB_TOKEN = "your_token"
python scripts/migrate_issues_to_static.py --apply --clean-title
```

Without `--apply`, the script only prints the planned changes.
