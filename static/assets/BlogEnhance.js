(function () {
  const STYLE_ID = "omnisyr-blog-enhance-style";

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      :root {
        --omni-accent: #006b75;
        --omni-accent-fg: #006b75;
        --omni-accent-emphasis: #0a565e;
        --omni-accent-soft: rgba(0, 107, 117, 0.10);
        --omni-accent-line: rgba(0, 107, 117, 0.32);
        --omni-radius: 8px;
        --omni-radius-sm: 6px;
        --omni-radius-lg: 8px;
        --omni-shadow-sm: 0 1px 2px rgba(31, 35, 40, 0.05), 0 4px 12px rgba(31, 35, 40, 0.04);
        --omni-shadow: 0 10px 28px rgba(31, 35, 40, 0.10);
        --omni-font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
        --omni-mono: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
        --omni-sticky-top: 84px;
      }

      html[data-color-mode="dark"] {
        --omni-accent: #3bc4cf;
        --omni-accent-fg: #5cd3dd;
        --omni-accent-emphasis: #8ce4ec;
        --omni-accent-soft: rgba(86, 211, 221, 0.13);
        --omni-accent-line: rgba(86, 211, 221, 0.30);
        --omni-shadow-sm: 0 1px 2px rgba(1, 4, 9, 0.55);
        --omni-shadow: 0 16px 42px rgba(1, 4, 9, 0.60);
      }

      @media (prefers-color-scheme: dark) {
        html[data-color-mode="auto"] {
          --omni-accent: #3bc4cf;
          --omni-accent-fg: #5cd3dd;
          --omni-accent-emphasis: #8ce4ec;
          --omni-accent-soft: rgba(86, 211, 221, 0.13);
          --omni-accent-line: rgba(86, 211, 221, 0.30);
          --omni-shadow-sm: 0 1px 2px rgba(1, 4, 9, 0.55);
          --omni-shadow: 0 16px 42px rgba(1, 4, 9, 0.60);
        }
      }

      html {
        scroll-padding-top: var(--omni-sticky-top);
        background: var(--color-canvas-default);
        color-scheme: light;
        -webkit-text-size-adjust: 100%;
      }

      html[data-color-mode="dark"] {
        color-scheme: dark;
      }

      @media (prefers-color-scheme: dark) {
        html[data-color-mode="auto"] {
          color-scheme: dark;
        }
      }

      html body {
        color: var(--fgColor-default, var(--color-fg-default)) !important;
        background-color: var(--color-canvas-default) !important;
        font-family: var(--omni-font) !important;
        font-size: 16px !important;
        line-height: 1.7 !important;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }

      body::before {
        content: none;
      }

      ::selection {
        background: rgba(0, 107, 117, 0.20);
      }

      a:focus-visible,
      button:focus-visible,
      [tabindex]:focus-visible {
        outline: 2px solid var(--omni-accent);
        outline-offset: 2px;
        border-radius: 4px;
      }

      .omnisyr-skip-link {
        position: fixed;
        z-index: 100;
        top: 8px;
        left: 8px;
        padding: 9px 12px;
        color: var(--fgColor-onEmphasis, #ffffff);
        background: var(--omni-accent-emphasis);
        border-radius: var(--omni-radius-sm);
        transform: translateY(-160%);
        transition: transform 0.16s ease;
      }

      .omnisyr-skip-link:focus {
        transform: translateY(0);
      }

      /* ---------- Header ---------- */
      #header {
        position: sticky;
        top: 0;
        z-index: 20;
        display: flex !important;
        align-items: center;
        gap: 16px;
        min-height: 60px;
        margin-bottom: 30px !important;
        padding: 10px 0 !important;
        border-bottom: 1px solid transparent !important;
        background: var(--color-canvas-default);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }

      #header.is-stuck {
        border-bottom-color: var(--borderColor-muted, var(--color-border-muted)) !important;
        box-shadow: 0 6px 22px rgba(31, 35, 40, 0.06);
      }

      html[data-color-mode="dark"] #header.is-stuck {
        box-shadow: 0 6px 22px rgba(1, 4, 9, 0.5);
      }

      .title-left {
        display: flex;
        min-width: 0;
        align-items: center;
        gap: 12px;
      }

      .avatar {
        width: 46px !important;
        height: 46px !important;
        flex: 0 0 auto;
        border: 1px solid var(--borderColor-muted, var(--color-border-muted));
        border-radius: 50% !important;
        box-shadow: var(--omni-shadow-sm);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease !important;
      }

      .avatar:hover {
        transform: translateY(-1px) scale(1.05) rotate(0deg) !important;
        box-shadow: var(--omni-shadow);
      }

      .title-left a,
      .blogTitle {
        margin-left: 0 !important;
        overflow: hidden;
        color: var(--fgColor-default, var(--color-fg-default)) !important;
        font-family: var(--omni-font) !important;
        font-size: 1.5rem !important;
        font-weight: 760 !important;
        letter-spacing: 0 !important;
        line-height: 1.1 !important;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .title-right {
        display: flex !important;
        flex: 0 0 auto;
        align-items: center;
        gap: 6px;
        margin: auto 0 auto auto !important;
      }

      .title-right .btn,
      .title-right button,
      #omnisyr-language-toggle {
        display: inline-grid !important;
        width: 38px;
        height: 38px;
        place-items: center;
        margin: 0 !important;
        padding: 0 !important;
        color: var(--fgColor-muted, var(--color-fg-muted)) !important;
        border-radius: var(--omni-radius-sm) !important;
        touch-action: manipulation;
        transition: background-color 0.16s ease, color 0.16s ease, transform 0.12s ease !important;
      }

      .title-right .btn:hover,
      .title-right button:hover,
      #omnisyr-language-toggle:hover {
        color: var(--omni-accent) !important;
        background: var(--omni-accent-soft) !important;
        text-decoration: none;
      }

      .title-right .btn:active,
      .title-right button:active,
      #omnisyr-language-toggle:active {
        transform: translateY(1px);
      }

      #content {
        min-width: 0;
      }

      /* ---------- Home: subtitle + summary ---------- */
      body > #content > div:first-child:not(.markdown-body):not(#taglabel) {
        max-width: 60ch;
        margin: 0 0 22px !important;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 1.02rem;
        line-height: 1.66;
      }

      .omnisyr-home-summary {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 0 0 22px;
        overflow: hidden;
        border: 1px solid var(--borderColor-muted, var(--color-border-muted));
        border-radius: var(--omni-radius);
        background: var(--color-canvas-subtle);
      }

      .omnisyr-home-summary__item {
        padding: 14px 18px;
        border-right: 1px solid var(--borderColor-muted, var(--color-border-muted));
      }

      .omnisyr-home-summary__item:last-child {
        border-right: 0;
      }

      .omnisyr-home-summary__label {
        display: block;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 0.78rem;
        line-height: 1.2;
      }

      .omnisyr-home-summary__value {
        display: block;
        margin-top: 6px;
        color: var(--fgColor-default, var(--color-fg-default));
        font-family: var(--omni-mono);
        font-size: 1.16rem;
        font-weight: 650;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
      }

      /* ---------- Post list ---------- */
      .SideNav {
        min-width: 0 !important;
        overflow: hidden;
        border: 1px solid var(--borderColor-muted, var(--color-border-muted)) !important;
        border-radius: var(--omni-radius) !important;
        background: var(--color-canvas-default);
        box-shadow: var(--omni-shadow-sm);
      }

      .SideNav-item {
        position: relative;
        padding: 15px 18px !important;
        gap: 16px;
        border-top: 0 !important;
        border-bottom: 1px solid var(--borderColor-muted, var(--color-border-muted));
        color: var(--fgColor-default, var(--color-fg-default)) !important;
        transition: background-color 0.16s ease !important;
      }

      .SideNav-item:last-child {
        border-bottom: 0;
      }

      .SideNav-item::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: var(--omni-accent);
        transform: scaleY(0);
        transform-origin: top;
        transition: transform 0.18s ease;
      }

      .SideNav-item:hover {
        background: var(--omni-accent-soft);
        text-decoration: none;
      }

      .SideNav-item:hover::before {
        transform: scaleY(1);
      }

      .SideNav-icon {
        flex: 0 0 auto;
        margin-right: 12px !important;
        color: var(--fgColor-muted, var(--color-fg-muted)) !important;
        transition: color 0.16s ease;
      }

      .SideNav-item:hover .SideNav-icon {
        color: var(--omni-accent) !important;
      }

      .listTitle {
        display: -webkit-box;
        max-width: 100%;
        overflow: hidden;
        color: inherit;
        font-size: 1.02rem;
        font-weight: 640;
        letter-spacing: 0;
        line-height: 1.45;
        text-overflow: ellipsis;
        white-space: normal !important;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        transition: color 0.16s ease;
      }

      .SideNav-item:hover .listTitle {
        color: var(--omni-accent-emphasis);
      }

      .listLabels {
        display: flex !important;
        flex: 0 0 auto;
        flex-wrap: wrap;
        align-items: center;
        justify-content: flex-end;
        gap: 6px;
        margin-left: 14px;
        white-space: normal !important;
      }

      /* The legacy theme wraps tag text in <object> and clamps it to 24px,
         which clips the label. Neutralise the wrapper so the pill sizes to text. */
      .Label object,
      .listLabels object {
        display: contents !important;
        max-width: none !important;
        max-height: none !important;
      }

      .Label {
        display: inline-flex !important;
        align-items: center;
        max-width: 100%;
        min-height: 22px;
        margin: 0 !important;
        padding: 2px 9px !important;
        border: 1px solid rgba(27, 31, 36, 0.06);
        border-radius: 999px !important;
        font-size: 0.74rem !important;
        font-weight: 600;
        line-height: 1.35;
        white-space: nowrap;
      }

      .Label a {
        color: inherit !important;
        text-decoration: none;
      }

      .LabelTime {
        font-family: var(--omni-mono);
        font-weight: 550;
        font-variant-numeric: tabular-nums;
        letter-spacing: 0;
      }

      /* ---------- Post / tag titles ---------- */
      .postTitle,
      .tagTitle {
        flex: 1 1 auto;
        min-width: 0;
        max-width: 760px;
        margin: auto 0 !important;
        overflow: hidden;
        color: var(--fgColor-default, var(--color-fg-default));
        font-family: var(--omni-font) !important;
        font-size: 2.1rem !important;
        font-weight: 760 !important;
        letter-spacing: 0 !important;
        line-height: 1.16 !important;
        text-overflow: ellipsis;
        text-wrap: balance;
      }

      /* ---------- Article body ---------- */
      .markdown-body {
        color: var(--fgColor-default, var(--color-fg-default));
        font-size: 1.02rem;
        line-height: 1.78;
      }

      #postBody {
        padding-bottom: 48px !important;
        border-bottom: 1px solid var(--borderColor-muted, var(--color-border-muted)) !important;
      }

      .markdown-body h1,
      .markdown-body h2,
      .markdown-body h3,
      .markdown-body h4,
      .markdown-body h5,
      .markdown-body h6 {
        scroll-margin-top: var(--omni-sticky-top);
        margin-top: 1.9em;
        color: var(--fgColor-default, var(--color-fg-default));
        font-family: var(--omni-font);
        font-weight: 720;
        letter-spacing: 0;
        line-height: 1.3;
        text-wrap: balance;
      }

      .markdown-body h1 { font-size: 1.7rem; }
      .markdown-body h2 {
        font-size: 1.38rem;
        padding-bottom: 0.3em;
        border-bottom: 1px solid var(--borderColor-muted, var(--color-border-muted));
      }
      .markdown-body h3 { font-size: 1.18rem; }
      .markdown-body h4 { font-size: 1.04rem; }

      .markdown-body p,
      .markdown-body li {
        line-height: 1.78;
        text-wrap: pretty;
      }

      .markdown-body a {
        color: var(--omni-accent-fg);
        text-decoration: underline;
        text-decoration-color: var(--omni-accent-line);
        text-decoration-thickness: 0.08em;
        text-underline-offset: 0.18em;
        transition: text-decoration-color 0.16s ease;
      }

      .markdown-body a:hover {
        text-decoration-color: var(--omni-accent-fg);
      }

      .markdown-body blockquote {
        margin-left: 0;
        padding: 0.2em 1em;
        color: var(--fgColor-muted, var(--color-fg-muted));
        border-left: 3px solid var(--omni-accent);
      }

      .markdown-body img {
        max-width: 100%;
        height: auto !important;
        border-radius: var(--omni-radius);
        background: var(--color-canvas-subtle);
        box-shadow: var(--omni-shadow-sm);
      }

      .markdown-body iframe {
        display: block;
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        aspect-ratio: 2 / 1;
      }

      .markdown-body :not(pre) > code {
        padding: 0.16em 0.4em;
        font-size: 0.88em;
        background: var(--omni-accent-soft);
        border-radius: var(--omni-radius-sm);
      }

      .markdown-body pre,
      .markdown-body .highlight pre {
        padding: 16px 18px;
        font-size: 0.8rem;
        line-height: 1.55;
        border: 1px solid var(--borderColor-muted, var(--color-border-muted));
        border-radius: var(--omni-radius) !important;
      }

      .markdown-body .snippet-clipboard-content {
        overflow-x: auto !important;
        overflow-y: hidden !important;
        overscroll-behavior-x: contain;
        overscroll-behavior-y: auto;
      }

      .markdown-body .snippet-clipboard-content pre {
        min-width: max-content;
        overflow: visible !important;
      }

      .markdown-body code,
      .markdown-body pre {
        font-family: var(--omni-mono) !important;
      }

      .markdown-body table {
        display: block;
        overflow-x: auto;
        border-radius: var(--omni-radius-sm);
      }

      .markdown-body hr {
        height: 1px;
        margin: 2.2em 0;
        background: var(--borderColor-muted, var(--color-border-muted));
        border: 0;
      }

      /* ---------- Tag page ---------- */
      .subnav-search {
        display: flex !important;
        width: min(360px, 48vw) !important;
        margin: 0 !important;
      }

      .subnav-search-input {
        width: 100% !important;
        min-width: 0;
        border-radius: var(--omni-radius-sm) !important;
      }

      #taglabel {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 20px !important;
      }

      #taglabel .Label[aria-pressed="true"] {
        outline: 2px solid var(--omni-accent);
        outline-offset: 2px;
      }

      /* ---------- Comments button + footer ---------- */
      #cmButton {
        min-height: 46px;
        margin-top: 32px !important;
        border-radius: var(--omni-radius) !important;
        font-weight: 650;
      }

      #footer {
        margin-top: 72px !important;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 0.86rem !important;
        line-height: 1.7;
      }

      #footer a {
        color: var(--omni-accent-fg);
      }

      /* ---------- Lightweight article tools ---------- */
      .markdown-body .omnisyr-heading-anchor {
        display: inline-grid;
        width: 30px;
        height: 30px;
        place-items: center;
        margin-left: 6px;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-family: var(--omni-font);
        font-size: 0.8em;
        font-weight: 500;
        line-height: 1;
        text-decoration: none;
        vertical-align: 0.05em;
        border-radius: var(--omni-radius-sm);
        opacity: 0;
        touch-action: manipulation;
        transition: color 0.16s ease, background-color 0.16s ease, opacity 0.16s ease;
      }

      .markdown-body h1:hover > .omnisyr-heading-anchor,
      .markdown-body h2:hover > .omnisyr-heading-anchor,
      .markdown-body h3:hover > .omnisyr-heading-anchor,
      .markdown-body h4:hover > .omnisyr-heading-anchor,
      .markdown-body .omnisyr-heading-anchor:focus-visible {
        opacity: 1;
      }

      .markdown-body .omnisyr-heading-anchor:hover {
        color: var(--omni-accent-fg);
        background: var(--omni-accent-soft);
      }

      .markdown-body img.omnisyr-zoomable-image {
        cursor: zoom-in;
      }

      .omnisyr-image-dialog {
        width: min(94vw, 1200px);
        max-width: none;
        max-height: 92dvh;
        margin: auto;
        padding: 0;
        overflow: hidden;
        color: var(--fgColor-default, var(--color-fg-default));
        background: var(--color-canvas-default);
        border: 0;
        border-radius: var(--omni-radius);
        box-shadow: var(--omni-shadow);
      }

      .omnisyr-image-dialog::backdrop {
        background: rgba(15, 23, 30, 0.78);
      }

      .omnisyr-image-dialog__close {
        position: absolute;
        top: 8px;
        right: 8px;
        z-index: 1;
        display: grid;
        width: 44px;
        height: 44px;
        place-items: center;
        padding: 0;
        color: var(--fgColor-default, var(--color-fg-default));
        font-size: 24px;
        line-height: 1;
        background: var(--color-canvas-default);
        border: 1px solid var(--borderColor-muted, var(--color-border-muted));
        border-radius: var(--omni-radius-sm);
        box-shadow: var(--omni-shadow-sm);
        cursor: pointer;
        touch-action: manipulation;
      }

      .omnisyr-image-dialog__figure {
        display: grid;
        max-height: 92dvh;
        margin: 0;
        overflow: auto;
      }

      .omnisyr-image-dialog__image {
        display: block;
        width: auto;
        max-width: 100%;
        height: auto;
        max-height: calc(92dvh - 54px);
        margin: auto;
        object-fit: contain;
      }

      .omnisyr-image-dialog__caption {
        min-height: 44px;
        padding: 12px 56px 12px 16px;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 0.88rem;
        line-height: 1.4;
      }

      .omnisyr-image-dialog__caption:empty {
        display: none;
      }

      @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
          scroll-behavior: auto !important;
          transition-duration: 0.01ms !important;
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
        }
      }

      @media (max-width: 720px) {
        #header {
          min-height: 56px;
          gap: 10px;
          margin-bottom: 22px !important;
          padding: 8px 0 !important;
        }

        .avatar {
          width: 40px !important;
          height: 40px !important;
        }

        .blogTitle {
          display: inline-block !important;
          max-width: calc(100vw - 210px);
          font-size: 1.12rem !important;
        }

        .title-right {
          gap: 2px;
        }

        .title-right .btn,
        .title-right button,
        #omnisyr-language-toggle {
          width: 36px;
          height: 36px;
        }

        body > #content > div:first-child:not(.markdown-body):not(#taglabel) {
          font-size: 0.97rem;
        }

        .omnisyr-home-summary__item {
          min-width: 0;
          padding: 11px 10px;
        }

        .omnisyr-home-summary__label {
          font-size: 0.7rem;
        }

        .omnisyr-home-summary__value {
          overflow: hidden;
          font-size: 0.92rem;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .SideNav-item {
          align-items: flex-start !important;
          flex-direction: column;
          gap: 10px;
          padding: 14px !important;
        }

        .SideNav-item > .d-flex {
          width: 100%;
        }

        .listLabels {
          justify-content: flex-start;
          margin-left: 28px;
        }

        .LabelTime {
          display: inline-flex !important;
        }

        .postTitle,
        .tagTitle {
          max-width: none;
          font-size: 1.45rem !important;
          white-space: normal;
        }

        .markdown-body .omnisyr-heading-anchor {
          opacity: 0.62;
        }

        .markdown-body {
          font-size: 1rem;
        }

        .subnav-search {
          width: min(100%, 220px) !important;
        }
      }

      @media (max-width: 480px) {
        .title-left a.blogTitle {
          max-width: none;
          font-size: 1.12rem !important;
        }

        #buttonRSS {
          display: none !important;
        }

        .title-right .btn,
        .title-right button,
        #omnisyr-language-toggle {
          width: 42px;
          height: 42px;
        }

        .subnav-search {
          width: 160px !important;
        }
      }
    `;

    document.head.appendChild(style);
  }

  function currentLanguage() {
    const explicit = document.documentElement.getAttribute("data-omnisyr-lang");
    if (explicit) return explicit.toLowerCase().startsWith("zh") ? "zh" : "en";

    try {
      const stored = localStorage.getItem("omnisyr_blog_lang");
      if (stored) return stored.toLowerCase().startsWith("zh") ? "zh" : "en";
    } catch (e) {
      // localStorage may be unavailable in strict privacy contexts.
    }

    const browserLangs = navigator.languages && navigator.languages.length
      ? navigator.languages
      : [navigator.language || navigator.userLanguage || ""];
    if (browserLangs.some(function (lang) { return String(lang).toLowerCase().startsWith("zh"); })) return "zh";

    const lang = document.documentElement.lang || "";
    return lang.toLowerCase().startsWith("zh") ? "zh" : "en";
  }

  function readableText(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }

  function uiText(key) {
    const isZh = currentLanguage() === "zh";
    const messages = {
      all: ["All", "全部"],
      tag: ["Tag", "标签"],
      search: ["Search", "搜索"],
      searchPosts: ["Search posts…", "搜索文章…"],
      noResults: ["No matching posts", "没有匹配的文章"],
      copyCode: ["Copy code", "复制代码"],
      copied: ["Copied", "已复制"]
    };
    const item = messages[key] || [key, key];
    return item[isZh ? 1 : 0];
  }

  // Pick black or white text for a given background so tag pills stay legible
  // regardless of the label colour set on GitHub.
  function readableTextColor(bg) {
    if (!bg) return null;
    let r;
    let g;
    let b;
    const hex = bg.trim().match(/^#?([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      let h = hex[1];
      if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
      r = parseInt(h.slice(0, 2), 16);
      g = parseInt(h.slice(2, 4), 16);
      b = parseInt(h.slice(4, 6), 16);
    } else {
      const nums = bg.match(/\d+(\.\d+)?/g);
      if (!nums || nums.length < 3) return null;
      r = parseFloat(nums[0]);
      g = parseFloat(nums[1]);
      b = parseFloat(nums[2]);
    }
    if ([r, g, b].some(function (v) { return isNaN(v); })) return null;
    function luminance(red, green, blue) {
      const channels = [red, green, blue].map(function (channel) {
        const value = channel / 255;
        return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
      });
      return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
    }

    const backgroundLuminance = luminance(r, g, b);
    const darkLuminance = luminance(31, 35, 40);
    const whiteContrast = 1.05 / (backgroundLuminance + 0.05);
    const darkContrast = (Math.max(backgroundLuminance, darkLuminance) + 0.05) /
      (Math.min(backgroundLuminance, darkLuminance) + 0.05);
    return darkContrast >= whiteContrast ? "#1f2328" : "#ffffff";
  }

  // Resolve the bilingual markers (mirrors Language.js) so a rebuilt label
  // never shows raw ;;;e / ;;;c / || tokens even if it runs before Language.js.
  function resolveMarkers(value, isZh) {
    let s = String(value || "");
    s = s.replace(/;;;e([\s\S]*?);;;e/g, isZh ? "" : "$1");
    s = s.replace(/;;;c([\s\S]*?);;;c/g, isZh ? "$1" : "");
    if (s.includes("||")) {
      const parts = s.split("||");
      if (parts.length === 2) s = isZh ? parts[1] : parts[0];
    }
    return s.replace(/\s+/g, " ").trim();
  }

  // Idempotent: records the background it last solved for, so repeated passes
  // (and Language.js re-renders) never fight over the colour.
  function setLabelContrast(label) {
    let bg = label.style.backgroundColor;
    if (!bg) {
      const m = (label.getAttribute("style") || "").match(/background-color:\s*([^;]+)/i);
      if (m) bg = m[1];
    }
    if (!bg) return;
    if (label.getAttribute("data-omni-c") === bg && label.style.color) return;
    const fg = readableTextColor(bg);
    if (!fg) return;
    label.style.color = fg;
    label.setAttribute("data-omni-c", bg);
  }

  // Resolve bilingual markers directly on text nodes, only touching nodes that
  // still contain markers (so it converges and never loops).
  function resolveTextMarkers(root, isZh) {
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    let node;
    while ((node = walker.nextNode())) nodes.push(node);
    nodes.forEach(function (text) {
      const value = text.nodeValue || "";
      if (value.indexOf(";;;") === -1 && value.indexOf("||") === -1) return;
      const resolved = resolveMarkers(value, isZh);
      if (resolved !== value) text.nodeValue = resolved;
    });
  }

  // Gmeek wraps each tag name in an <object> (a parser trick so a tag <a> can
  // nest inside the row <a>). Chromium renders that <object> empty, and any
  // re-parse auto-corrects the invalid nested anchor and corrupts the row.
  // Rebuild the pill into a plain <span> (inside list rows) or a real <a>
  // (standalone tag page) so it sizes to its text and survives re-parsing.
  function processLabel(label, isZh) {
    const obj = label.querySelector("object");
    if (obj) {
      const innerA = obj.querySelector("a");
      const text = resolveMarkers(innerA ? innerA.textContent : obj.textContent, isZh);
      const href = innerA ? innerA.getAttribute("href") : null;
      const nested = !!label.parentElement && !!label.parentElement.closest("a");

      let el;
      if (href && !nested) {
        el = document.createElement("a");
        el.setAttribute("href", resolveMarkers(href, isZh));
      } else {
        el = document.createElement("span");
      }
      el.textContent = text;

      while (label.firstChild) label.removeChild(label.firstChild);
      label.appendChild(el);
    } else {
      resolveTextMarkers(label, isZh);
    }
    setLabelContrast(label);
  }

  // One idempotent pass over labels + bilingual titles. Safe to run repeatedly.
  function processContent() {
    const isZh = currentLanguage() === "zh";
    document.querySelectorAll(".Label").forEach(function (label) {
      processLabel(label, isZh);
    });
    document.querySelectorAll(".listTitle, .tagTitle").forEach(function (el) {
      resolveTextMarkers(el, isZh);
    });
    installTagPageBehavior();
  }

  function setText(element, value) {
    if (element && element.textContent !== value) element.textContent = value;
  }

  function decodeHash(value) {
    try {
      return decodeURIComponent(value || "");
    } catch (e) {
      return value || "";
    }
  }

  function tagSlug(label) {
    if (label === "All") return "all";
    let source = resolveMarkers(label, false) || resolveMarkers(label, true) || label;
    if (source.normalize) source = source.normalize("NFKD");
    const slug = source
      .toLowerCase()
      .replace(/&/g, " and ")
      .replace(/['']/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return slug || "tag";
  }

  function createTagState() {
    if (!window.jsonData || typeof window.jsonData !== "object" || !Array.isArray(window.tagList)) return null;

    const labels = window.tagList.slice();
    const labelToSlug = Object.create(null);
    const slugToLabel = Object.create(null);

    labels.forEach(function (label) {
      const base = tagSlug(label);
      let slug = base;
      let suffix = 2;
      while (slugToLabel[slug] && slugToLabel[slug] !== label) {
        slug = base + "-" + suffix;
        suffix++;
      }
      labelToSlug[label] = slug;
      slugToLabel[slug] = label;
    });

    const posts = Object.keys(window.jsonData)
      .filter(function (key) {
        return key !== "labelColorDict" && window.jsonData[key] && Array.isArray(window.jsonData[key].labels);
      })
      .map(function (key) {
        return window.jsonData[key];
      });

    return { labels: labels, labelToSlug: labelToSlug, slugToLabel: slugToLabel, posts: posts };
  }

  function resolveTagInput(input, state) {
    const value = decodeHash(String(input || "").replace(/^#/, "")).trim();
    if (!value || value.toLowerCase() === "all") return "All";
    if (state.slugToLabel[value]) return state.slugToLabel[value];
    if (state.labelToSlug[value]) return value;

    const lowered = value.toLowerCase();
    if (state.slugToLabel[lowered]) return state.slugToLabel[lowered];

    const isZh = currentLanguage() === "zh";
    for (let i = 0; i < state.labels.length; i++) {
      const label = state.labels[i];
      if (
        resolveMarkers(label, isZh) === value ||
        resolveMarkers(label, false) === value ||
        resolveMarkers(label, true) === value
      ) {
        return label;
      }
    }

    return value;
  }

  function tagDisplayName(label) {
    return label === "All" ? uiText("all") : resolveMarkers(label, currentLanguage() === "zh");
  }

  function setNotFound(message, show) {
    const notFind = document.getElementsByClassName("notFind")[0];
    if (!notFind) return;
    notFind.setAttribute("role", "status");
    notFind.setAttribute("aria-live", "polite");
    notFind.style.display = show ? "block" : "none";
    if (message) notFind.textContent = message;
  }

  function applyTagSearch(query, state) {
    const lists = Array.from(document.getElementsByClassName("lists"));
    const tagTitle = document.getElementsByClassName("tagTitle")[0];
    const input = document.getElementsByClassName("subnav-search-input")[0];
    const searchInput = String(query || "");
    const needle = searchInput.toUpperCase();
    let count = 0;

    if (input && input.value !== searchInput) input.value = searchInput;
    const searchLabel = uiText("search");
    setText(tagTitle, searchInput ? searchLabel + " #" + searchInput : searchLabel);
    document.title = (searchInput || searchLabel) + " - OmnisyR's Blog";

    lists.forEach(function (list, index) {
      const post = state.posts[index];
      const haystack = [
        list.textContent,
        post && post.postTitle,
        post && post.labels && post.labels.join(" "),
        post && post.labels && post.labels.map(function (label) {
          return resolveMarkers(label, false) + " " + resolveMarkers(label, true);
        }).join(" ")
      ].join(" ").toUpperCase();

      const visible = !needle || haystack.indexOf(needle) !== -1;
      list.style.display = visible ? "block" : "none";
      if (visible) count++;
    });

    setNotFound(uiText("noResults") + (searchInput ? ': "' + searchInput + '"' : ""), count === 0);
  }

  function applyTagFilter(label, state) {
    const lists = Array.from(document.getElementsByClassName("lists"));
    const tagTitle = document.getElementsByClassName("tagTitle")[0];
    const input = document.getElementsByClassName("subnav-search-input")[0];
    const display = tagDisplayName(label);
    let count = 0;

    if (input) input.value = "";
    setText(tagTitle, uiText("tag") + " #" + display);
    document.title = display + " - OmnisyR's Blog";

    lists.forEach(function (list, index) {
      const post = state.posts[index];
      const visible = label === "All" || !!(post && post.labels && post.labels.indexOf(label) !== -1);
      list.style.display = visible ? "block" : "none";
      if (visible) count++;
    });

    setNotFound(uiText("noResults") + ': "' + display + '"', count === 0);

    document.querySelectorAll("#taglabel > .Label").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.getAttribute("data-omni-tag") === state.labelToSlug[label]));
    });
  }

  function installTagPageBehavior() {
    if (!document.querySelector(".tagTitle") || !document.getElementById("taglabel")) return;

    const state = createTagState();
    if (!state || !state.labels.length || !document.querySelector(".SideNav .lists")) return;
    window.__omniTagState = state;

    Array.from(document.querySelectorAll("#taglabel > .Label")).forEach(function (button, index) {
      const label = state.labels[index];
      if (!label) return;
      button.removeAttribute("onclick");
      button.setAttribute("type", "button");
      button.setAttribute("data-omni-tag", state.labelToSlug[label]);
      button.setAttribute("aria-pressed", "false");
      button.onclick = function (event) {
        event.preventDefault();
        window.updateShowTag(label);
      };
    });

    Array.from(document.querySelectorAll(".SideNav .lists")).forEach(function (list, index) {
      const post = state.posts[index];
      const slugs = post && post.labels
        ? post.labels.map(function (label) { return state.labelToSlug[label]; }).filter(Boolean)
        : [];
      list.setAttribute("data-omni-tags", slugs.join(" "));
    });

    window.updateShowTag = function (labelOrSlug) {
      const activeState = createTagState() || state;
      const label = resolveTagInput(labelOrSlug, activeState);
      const slug = activeState.labelToSlug[label];
      if (slug) window.location.hash = "#" + encodeURIComponent(slug);
      applyTagFilter(label, activeState);
    };

    window.setClassDisplay = function (labelOrSlug) {
      const activeState = createTagState() || state;
      const label = resolveTagInput(labelOrSlug, activeState);
      if (activeState.labelToSlug[label]) {
        applyTagFilter(label, activeState);
      } else {
        applyTagSearch(label, activeState);
      }
    };

    window.searchShow = function () {
      const input = document.getElementsByClassName("subnav-search-input")[0];
      const query = input ? input.value : "";
      if (query) window.location.hash = "#" + encodeURIComponent(query);
      else history.replaceState(null, "", window.location.pathname);
      applyTagSearch(query, createTagState() || state);
    };

    if (!window.__omniTagHashListener) {
      window.__omniTagHashListener = true;
      window.addEventListener("hashchange", function () {
        if (document.querySelector(".tagTitle") && window.setClassDisplay) {
          window.setClassDisplay(decodeHash(window.location.hash.slice(1)) || "All");
        }
      });
    }

    const searchInput = document.getElementsByClassName("subnav-search-input")[0];
    const searchButton = document.querySelector(".subnav-search button");
    if (searchInput && !searchInput.hasAttribute("data-omni-search")) {
      searchInput.setAttribute("data-omni-search", "ready");
      searchInput.setAttribute("name", "q");
      searchInput.setAttribute("autocomplete", "off");
      searchInput.setAttribute("spellcheck", "false");
      searchInput.setAttribute("placeholder", uiText("searchPosts"));
      searchInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
          event.preventDefault();
          window.searchShow();
        }
      });
    }
    if (searchButton && !searchButton.hasAttribute("data-omni-search")) {
      searchButton.setAttribute("data-omni-search", "ready");
      searchButton.removeAttribute("onclick");
      searchButton.addEventListener("click", function () {
        window.searchShow();
      });
    }
    if (document.activeElement !== searchInput) {
      window.setClassDisplay(decodeHash(window.location.hash.slice(1)) || "All");
    }
  }

  // The tag/search page builds its chips and list after the initial scripts run.
  // A short debounce keeps that dynamic content marker-free without observing
  // MathJax and other mutation-heavy article pages.
  function watchContent() {
    const target = document.getElementById("content");
    if (!target || !document.querySelector(".tagTitle") || !window.MutationObserver) return;
    let timer = null;
    const observer = new MutationObserver(function () {
      if (timer) return;
      timer = window.setTimeout(function () {
        timer = null;
        processContent();
      }, 40);
    });
    observer.observe(target, { childList: true, subtree: true });
  }

  function enhanceExternalLinks() {
    document.querySelectorAll('a[target="_blank"]').forEach(function (link) {
      const rel = new Set(readableText(link.getAttribute("rel")).split(" ").filter(Boolean));
      rel.add("noopener");
      rel.add("noreferrer");
      link.setAttribute("rel", Array.from(rel).join(" "));
    });
  }

  function enhanceSemantics() {
    const content = document.getElementById("content");
    const header = document.getElementById("header");
    const footer = document.getElementById("footer");

    if (content) {
      content.setAttribute("role", "main");
      content.setAttribute("tabindex", "-1");
    }
    if (header) header.setAttribute("role", "banner");
    if (footer) footer.setAttribute("role", "contentinfo");

    const brandLink = document.querySelector("a.blogTitle:not([href])");
    if (brandLink) brandLink.setAttribute("href", "/");

    if (content && !document.querySelector(".omnisyr-skip-link")) {
      const skipLink = document.createElement("a");
      skipLink.className = "omnisyr-skip-link";
      skipLink.href = "#content";
      skipLink.textContent = currentLanguage() === "zh" ? "跳到正文" : "Skip to content";
      document.body.prepend(skipLink);
    }

    const themeLink = document.querySelector('#header a[onclick*="modeSwitch"]');
    if (themeLink) {
      const button = document.createElement("button");
      button.id = "omnisyr-theme-toggle";
      button.type = "button";
      button.className = themeLink.className;
      button.title = themeLink.title;
      button.setAttribute("aria-label", themeLink.getAttribute("aria-label") || themeLink.title);
      button.innerHTML = themeLink.innerHTML;
      button.addEventListener("click", function () { window.modeSwitch(); });
      themeLink.replaceWith(button);
    }
  }

  function enhanceIconButtons() {
    document.querySelectorAll("#header a, #header button").forEach(function (item) {
      if (item.getAttribute("aria-label")) return;
      const title = item.getAttribute("title");
      if (title) item.setAttribute("aria-label", title);
    });
    document.querySelectorAll("#header svg").forEach(function (icon) {
      icon.setAttribute("aria-hidden", "true");
      icon.setAttribute("focusable", "false");
    });
  }

  function enhanceCodeCopyButtons() {
    document.querySelectorAll("clipboard-copy[role='button']").forEach(function (button) {
      if (button.hasAttribute("data-omni-copy")) return;
      button.setAttribute("data-omni-copy", "ready");
      button.setAttribute("tabindex", "0");
      button.setAttribute("aria-label", uiText("copyCode"));
      button.setAttribute("title", uiText("copyCode"));
      button.querySelectorAll("svg").forEach(function (icon) {
        icon.setAttribute("aria-hidden", "true");
        icon.setAttribute("focusable", "false");
      });
      button.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          button.click();
        }
      });
    });

    document.querySelectorAll(".copy-feedback").forEach(function (feedback) {
      feedback.textContent = uiText("copied") + "!";
      feedback.setAttribute("role", "status");
      feedback.setAttribute("aria-live", "polite");
    });
  }

  function enhanceImages() {
    let priorityAssigned = false;
    document.querySelectorAll(".markdown-body img").forEach(function (image) {
      const nearViewport = image.getBoundingClientRect().top < window.innerHeight * 1.5;
      if (!image.getAttribute("loading")) image.setAttribute("loading", nearViewport ? "eager" : "lazy");
      if (image.getAttribute("fetchpriority") === "high") priorityAssigned = true;
      if (nearViewport && !priorityAssigned && !image.getAttribute("fetchpriority")) {
        image.setAttribute("fetchpriority", "high");
        priorityAssigned = true;
      }
      if (!image.getAttribute("decoding")) image.setAttribute("decoding", "async");

      if (!image.hasAttribute("alt") || !image.getAttribute("alt")) {
        let label = "";
        try {
          const filename = decodeURIComponent(new URL(image.currentSrc || image.src, location.href).pathname.split("/").pop() || "");
          label = filename.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ").trim();
        } catch (error) {
          label = "";
        }
        image.alt = label || (currentLanguage() === "zh" ? "文章插图" : "Article figure");
      }
    });
  }

  // Toggle a shadow on the sticky header once the page scrolls, using an
  // IntersectionObserver sentinel (no scroll listeners).
  function setupStickyHeader() {
    const header = document.getElementById("header");
    if (!header || !("IntersectionObserver" in window)) return;
    if (document.getElementById("omnisyr-sticky-sentinel")) return;

    const sentinel = document.createElement("div");
    sentinel.id = "omnisyr-sticky-sentinel";
    sentinel.setAttribute("aria-hidden", "true");
    sentinel.style.cssText = "position:absolute;top:0;left:0;width:1px;height:1px;pointer-events:none;";
    document.body.prepend(sentinel);

    const observer = new IntersectionObserver(function (entries) {
      header.classList.toggle("is-stuck", !entries[0].isIntersecting);
    }, { threshold: 0 });
    observer.observe(sentinel);
  }

  function createHomeSummary() {
    const content = document.getElementById("content");
    const list = content && content.querySelector(":scope > nav.SideNav");
    if (!content || !list || document.querySelector(".postTitle") || document.querySelector(".tagTitle")) return;
    if (document.querySelector(".omnisyr-home-summary")) return;

    const posts = Array.from(list.querySelectorAll(".SideNav-item"));
    const labels = new Set();
    posts.forEach(function (post) {
      post.querySelectorAll(".LabelName").forEach(function (label) {
        labels.add(readableText(label.textContent));
      });
    });

    const latestDate = posts
      .map(function (post) {
        const date = post.querySelector(".LabelTime");
        return readableText(date && date.textContent);
      })
      .filter(function (date) { return /^\d{4}-\d{2}-\d{2}$/.test(date); })
      .sort()
      .pop() || "";

    const isZh = currentLanguage() === "zh";
    const copy = isZh
      ? { posts: "文章", topics: "主题", latest: "最新" }
      : { posts: "Posts", topics: "Topics", latest: "Latest" };

    const summary = document.createElement("section");
    summary.className = "omnisyr-home-summary";
    summary.setAttribute("aria-label", isZh ? "博客摘要" : "Blog summary");

    [
      [copy.posts, String(posts.length)],
      [copy.topics, String(labels.size)],
      [copy.latest, latestDate || "-"]
    ].forEach(function (item) {
      const stat = document.createElement("div");
      stat.className = "omnisyr-home-summary__item";
      stat.innerHTML =
        '<span class="omnisyr-home-summary__label"></span>' +
        '<span class="omnisyr-home-summary__value"></span>';
      stat.children[0].textContent = item[0];
      stat.children[1].textContent = item[1];
      summary.appendChild(stat);
    });

    content.insertBefore(summary, list);
  }

  function enhance() {
    processContent();
    createHomeSummary();
    enhanceExternalLinks();
    enhanceSemantics();
    enhanceIconButtons();
    enhanceCodeCopyButtons();
    enhanceImages();
    setupStickyHeader();
    watchContent();
  }

  injectStyle();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhance);
  } else {
    enhance();
  }

  if (document.readyState === "complete") {
    enhanceCodeCopyButtons();
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      window.setTimeout(enhanceCodeCopyButtons, 0);
    }, { once: true });
  }
})();
