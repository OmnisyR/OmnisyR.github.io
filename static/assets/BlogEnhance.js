(function () {
  const STYLE_ID = "omnisyr-blog-enhance-style";

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      :root {
        --omni-accent: #006b75;
        --omni-accent-soft: rgba(0, 107, 117, 0.12);
        --omni-radius: 8px;
        --omni-shadow: 0 18px 45px rgba(31, 35, 40, 0.08);
        --omni-font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
        --omni-mono: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
      }

      html {
        scroll-padding-top: 96px;
        background: var(--color-canvas-default);
      }

      html body {
        width: min(100% - 32px, 1040px) !important;
        min-width: 0 !important;
        max-width: none !important;
        margin: 0 auto !important;
        padding: 32px 0 56px !important;
        font-family: var(--omni-font) !important;
        font-size: 16px !important;
        line-height: 1.68 !important;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
      }

      body::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        background:
          linear-gradient(180deg, rgba(0, 107, 117, 0.06), transparent 320px),
          radial-gradient(circle at 90% 0%, rgba(9, 105, 218, 0.08), transparent 28rem);
      }

      #header {
        position: sticky;
        top: 0;
        z-index: 20;
        display: flex !important;
        align-items: center;
        gap: 16px;
        min-height: 64px;
        margin-bottom: 32px !important;
        padding: 8px 0 !important;
        border-bottom: 1px solid var(--borderColor-muted, var(--color-border-muted)) !important;
        background: var(--color-canvas-default);
        background: color-mix(in srgb, var(--color-canvas-default) 88%, transparent);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
      }

      .title-left {
        display: flex;
        min-width: 0;
        align-items: center;
        gap: 12px;
      }

      .avatar {
        width: 56px !important;
        height: 56px !important;
        flex: 0 0 auto;
        border: 1px solid var(--borderColor-muted, var(--color-border-muted));
        border-radius: 50% !important;
        box-shadow: 0 8px 26px rgba(31, 35, 40, 0.12);
        transition: transform 180ms ease, box-shadow 180ms ease !important;
      }

      .avatar:hover {
        transform: translateY(-1px) scale(1.04) !important;
        box-shadow: 0 12px 30px rgba(31, 35, 40, 0.16);
      }

      .title-left a,
      .blogTitle {
        margin-left: 0 !important;
        overflow: hidden;
        color: var(--fgColor-default, var(--color-fg-default)) !important;
        font-family: var(--omni-font) !important;
        font-size: clamp(1.35rem, 3vw, 2.15rem) !important;
        font-weight: 750 !important;
        letter-spacing: 0 !important;
        line-height: 1.1 !important;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .title-right {
        display: flex !important;
        flex: 0 0 auto;
        align-items: center;
        gap: 8px;
        margin: auto 0 auto auto !important;
      }

      .title-right .btn,
      .title-right button,
      #omnisyr-language-toggle {
        display: inline-grid !important;
        width: 40px;
        height: 40px;
        place-items: center;
        margin: 0 !important;
        padding: 0 !important;
        border-radius: var(--omni-radius) !important;
        transition: background-color 160ms ease, border-color 160ms ease, transform 160ms ease !important;
      }

      .title-right .btn:hover,
      .title-right button:hover,
      #omnisyr-language-toggle:hover {
        background: var(--omni-accent-soft) !important;
        border-color: rgba(0, 107, 117, 0.28) !important;
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

      body > #content > div:first-child:not(.markdown-body):not(#taglabel) {
        max-width: 68ch;
        margin: 0 0 24px !important;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 1.05rem;
        line-height: 1.72;
      }

      .omnisyr-home-summary {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 0 0 18px;
        overflow: hidden;
        border: 1px solid var(--borderColor-muted, var(--color-border-muted));
        border-radius: var(--omni-radius);
        background: var(--color-canvas-subtle);
      }

      .omnisyr-home-summary__item {
        padding: 14px 16px;
        border-right: 1px solid var(--borderColor-muted, var(--color-border-muted));
      }

      .omnisyr-home-summary__item:last-child {
        border-right: 0;
      }

      .omnisyr-home-summary__label {
        display: block;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 0.76rem;
        line-height: 1.2;
      }

      .omnisyr-home-summary__value {
        display: block;
        margin-top: 4px;
        color: var(--fgColor-default, var(--color-fg-default));
        font-family: var(--omni-mono);
        font-size: 1.05rem;
        font-weight: 650;
        line-height: 1.25;
      }

      .SideNav {
        min-width: 0 !important;
        overflow: hidden;
        border: 1px solid var(--borderColor-muted, var(--color-border-muted)) !important;
        border-radius: var(--omni-radius) !important;
        background: var(--color-canvas-default);
        box-shadow: var(--omni-shadow);
      }

      .SideNav-item {
        min-height: 68px;
        padding: 16px 18px !important;
        gap: 16px;
        border-top: 0 !important;
        border-bottom: 1px solid var(--borderColor-muted, var(--color-border-muted));
        color: var(--fgColor-default, var(--color-fg-default)) !important;
        transition: background-color 160ms ease, transform 160ms ease !important;
      }

      .SideNav-item:last-child {
        border-bottom: 0;
      }

      .SideNav-item:hover {
        background: var(--color-accent-subtle);
        text-decoration: none;
        transform: translateX(2px);
      }

      .SideNav-icon {
        flex: 0 0 auto;
        margin-right: 14px !important;
        color: var(--omni-accent);
      }

      .listTitle {
        display: -webkit-box;
        max-width: 100%;
        overflow: hidden;
        color: inherit;
        font-size: 1rem;
        font-weight: 650;
        line-height: 1.45;
        text-overflow: ellipsis;
        white-space: normal !important;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
      }

      .listLabels {
        display: flex !important;
        flex: 0 0 auto;
        flex-wrap: wrap;
        align-items: center;
        justify-content: flex-end;
        gap: 6px;
        margin-left: 12px;
        white-space: normal !important;
      }

      .Label {
        display: inline-flex !important;
        align-items: center;
        max-width: 100%;
        min-height: 24px;
        margin: 0 !important;
        padding: 3px 8px !important;
        border-radius: 999px !important;
        color: #fff !important;
        font-size: 0.76rem !important;
        font-weight: 650;
        line-height: 1.35;
      }

      .Label a {
        color: inherit !important;
      }

      .LabelTime {
        font-family: var(--omni-mono);
      }

      .postTitle,
      .tagTitle {
        max-width: min(760px, calc(100vw - 280px));
        margin: auto 0 !important;
        overflow: hidden;
        color: var(--fgColor-default, var(--color-fg-default));
        font-family: var(--omni-font) !important;
        font-size: clamp(1.55rem, 3.8vw, 2.55rem) !important;
        font-weight: 760 !important;
        letter-spacing: 0 !important;
        line-height: 1.14 !important;
        text-overflow: ellipsis;
      }

      .markdown-body {
        max-width: 78ch;
        margin: 0 auto;
        color: var(--fgColor-default, var(--color-fg-default));
        font-size: 1.02rem;
        line-height: 1.82;
      }

      #postBody {
        padding-bottom: 48px !important;
        border-bottom: 1px solid var(--borderColor-muted, var(--color-border-muted)) !important;
      }

      .markdown-body h1,
      .markdown-body h2,
      .markdown-body h3,
      .markdown-body h4 {
        scroll-margin-top: 96px;
        color: var(--fgColor-default, var(--color-fg-default));
        font-family: var(--omni-font);
        font-weight: 740;
        letter-spacing: 0;
        line-height: 1.25;
      }

      .markdown-body p,
      .markdown-body li {
        line-height: 1.82;
      }

      .markdown-body a {
        text-decoration-thickness: 0.08em;
        text-underline-offset: 0.18em;
      }

      .markdown-body blockquote {
        color: var(--fgColor-muted, var(--color-fg-muted));
        border-left-color: var(--omni-accent);
      }

      .markdown-body img {
        max-width: 100%;
        height: auto !important;
        border-radius: var(--omni-radius);
        background: var(--color-canvas-subtle);
        box-shadow: 0 12px 34px rgba(31, 35, 40, 0.1);
      }

      .markdown-body pre {
        border-radius: var(--omni-radius);
        padding: 18px;
        line-height: 1.6;
      }

      .markdown-body code,
      .markdown-body pre {
        font-family: var(--omni-mono) !important;
      }

      .markdown-body table {
        display: block;
        overflow-x: auto;
        border-radius: var(--omni-radius);
      }

      .subnav-search {
        display: flex !important;
        width: min(360px, 48vw) !important;
        margin: 0 !important;
      }

      .subnav-search-input {
        width: 100% !important;
        min-width: 0;
      }

      #taglabel {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 18px !important;
      }

      #cmButton {
        min-height: 46px;
        margin-top: 32px !important;
        border-radius: var(--omni-radius) !important;
        font-weight: 650;
      }

      #footer {
        margin-top: 72px !important;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 0.88rem !important;
        line-height: 1.7;
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
          width: min(100% - 24px, 1040px) !important;
          padding: 16px 0 40px !important;
        }

        #header {
          min-height: 64px;
          gap: 10px;
          margin-bottom: 22px !important;
          padding: 10px 0 !important;
        }

        .avatar {
          width: 42px !important;
          height: 42px !important;
        }

        .blogTitle {
          display: inline-block !important;
          max-width: calc(100vw - 218px);
          font-size: 1.18rem !important;
        }

        .title-right {
          gap: 4px;
        }

        .title-right .btn,
        .title-right button,
        #omnisyr-language-toggle {
          width: 36px;
          height: 36px;
        }

        body > #content > div:first-child:not(.markdown-body):not(#taglabel) {
          font-size: 0.98rem;
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
          margin-left: 30px;
        }

        .LabelTime {
          display: inline-flex !important;
        }

        .postTitle,
        .tagTitle {
          max-width: none;
          font-size: 1.35rem !important;
          white-space: normal;
        }

        .subnav-search {
          width: min(100%, 220px) !important;
        }
      }

      @media (max-width: 480px) {
        .blogTitle {
          max-width: 38vw;
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
      ? { posts: "\u6587\u7ae0", topics: "\u4e3b\u9898", latest: "\u6700\u65b0" }
      : { posts: "Posts", topics: "Topics", latest: "Latest" };

    const summary = document.createElement("section");
    summary.className = "omnisyr-home-summary";
    summary.setAttribute("aria-label", isZh ? "\u535a\u5ba2\u6458\u8981" : "Blog summary");

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
    createHomeSummary();
    enhanceExternalLinks();
    enhanceIconButtons();
    enhanceImages();
  }

  injectStyle();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhance);
  } else {
    enhance();
  }
})();
