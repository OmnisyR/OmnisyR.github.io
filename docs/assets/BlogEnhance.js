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
        --omni-radius: 10px;
        --omni-radius-sm: 7px;
        --omni-radius-lg: 16px;
        --omni-shadow-sm: 0 1px 2px rgba(31, 35, 40, 0.06), 0 2px 8px rgba(31, 35, 40, 0.05);
        --omni-shadow: 0 14px 38px rgba(31, 35, 40, 0.10);
        --omni-font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
        --omni-mono: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
        --omni-maxw: 1000px;
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
        -webkit-text-size-adjust: 100%;
      }

      html body {
        width: min(100% - 40px, var(--omni-maxw)) !important;
        min-width: 0 !important;
        max-width: none !important;
        margin: 0 auto !important;
        padding: 26px 0 72px !important;
        color: var(--fgColor-default, var(--color-fg-default)) !important;
        font-family: var(--omni-font) !important;
        font-size: 16px !important;
        line-height: 1.7 !important;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }

      body::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        background: radial-gradient(120% 56% at 50% -8%, var(--omni-accent-soft), transparent 62%);
      }

      ::selection {
        background: color-mix(in srgb, var(--omni-accent) 24%, transparent);
      }

      a:focus-visible,
      button:focus-visible,
      [tabindex]:focus-visible {
        outline: 2px solid var(--omni-accent);
        outline-offset: 2px;
        border-radius: 4px;
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
        background: color-mix(in srgb, var(--color-canvas-default) 80%, transparent);
        backdrop-filter: saturate(180%) blur(14px);
        -webkit-backdrop-filter: saturate(180%) blur(14px);
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
        font-size: clamp(1.22rem, 2.6vw, 1.7rem) !important;
        font-weight: 760 !important;
        letter-spacing: -0.01em !important;
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
        letter-spacing: -0.005em;
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
        letter-spacing: -0.01em;
      }

      /* ---------- Post / tag titles ---------- */
      .postTitle,
      .tagTitle {
        max-width: min(760px, calc(100vw - 280px));
        margin: auto 0 !important;
        overflow: hidden;
        color: var(--fgColor-default, var(--color-fg-default));
        font-family: var(--omni-font) !important;
        font-size: clamp(1.55rem, 3.6vw, 2.4rem) !important;
        font-weight: 760 !important;
        letter-spacing: -0.018em !important;
        line-height: 1.16 !important;
        text-overflow: ellipsis;
      }

      /* ---------- Article body ---------- */
      .markdown-body {
        max-width: 72ch;
        margin: 0 auto;
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
        letter-spacing: -0.012em;
        line-height: 1.3;
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

      .markdown-body :not(pre) > code {
        padding: 0.16em 0.4em;
        font-size: 0.88em;
        background: var(--omni-accent-soft);
        border-radius: var(--omni-radius-sm);
      }

      .markdown-body pre,
      .markdown-body .highlight pre {
        padding: 16px 18px;
        line-height: 1.6;
        border: 1px solid var(--borderColor-muted, var(--color-border-muted));
        border-radius: var(--omni-radius) !important;
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
        html body {
          width: min(100% - 24px, var(--omni-maxw)) !important;
          padding: 14px 0 44px !important;
        }

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

        .omnisyr-home-summary {
          grid-template-columns: 1fr;
        }

        .omnisyr-home-summary__item {
          border-right: 0;
          border-bottom: 1px solid var(--borderColor-muted, var(--color-border-muted));
        }

        .omnisyr-home-summary__item:last-child {
          border-bottom: 0;
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

        .markdown-body {
          font-size: 1rem;
        }

        .subnav-search {
          width: min(100%, 220px) !important;
        }
      }

      @media (max-width: 480px) {
        .blogTitle {
          max-width: 40vw;
        }

        .subnav-search {
          width: 160px !important;
        }
      }
    `;

    document.head.appendChild(style);
  }

  function currentLanguage() {
    const lang = document.documentElement.lang || "";
    return lang.toLowerCase().startsWith("zh") ? "zh" : "en";
  }

  function readableText(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
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
    const yiq = (r * 299 + g * 587 + b * 114) / 1000;
    return yiq >= 146 ? "#1f2328" : "#ffffff";
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
  }

  // The tag/search page builds its chips and list from postList.json after
  // load. Language.js retranslates that via requestAnimationFrame, which is
  // throttled in background tabs; a short setTimeout debounce keeps labels
  // legible and marker-free regardless. Idempotent, so it converges quickly.
  function watchContent() {
    const target = document.getElementById("content");
    if (!target || !window.MutationObserver) return;
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

  function enhanceIconButtons() {
    document.querySelectorAll("#header a, #header button").forEach(function (item) {
      if (item.getAttribute("aria-label")) return;
      const title = item.getAttribute("title");
      if (title) item.setAttribute("aria-label", title);
    });
  }

  function enhanceImages() {
    document.querySelectorAll(".markdown-body img").forEach(function (image) {
      if (!image.getAttribute("loading")) image.setAttribute("loading", "lazy");
      if (!image.getAttribute("decoding")) image.setAttribute("decoding", "async");
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

    const latestDate = posts.length
      ? readableText(posts[0].querySelector(".LabelTime") && posts[0].querySelector(".LabelTime").textContent)
      : "";

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
    enhanceIconButtons();
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
})();
