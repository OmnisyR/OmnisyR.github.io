(function () {
  "use strict";

  const denotes = new Map();
  const renderedDenotes = new Map();
  let activeDenote = "";

  function currentLanguage() {
    if (window.OmnisyrLanguage) return window.OmnisyrLanguage.get();
    return document.documentElement.lang.toLowerCase().startsWith("zh") ? "zh" : "en";
  }

  function message(english, chinese) {
    return currentLanguage() === "zh" ? chinese : english;
  }

  function resolveLegacy(value) {
    if (window.OmnisyrLanguage) return window.OmnisyrLanguage.resolveLegacy(value);
    const isZh = currentLanguage() === "zh";
    return String(value || "")
      .replace(/;;;e([\s\S]*?);;;e/gm, isZh ? "" : "$1")
      .replace(/;;;c([\s\S]*?);;;c/gm, isZh ? "$1" : "");
  }

  function htmlText(value) {
    const template = document.createElement("template");
    template.innerHTML = value;
    return template.content.textContent.replace(/\s+/g, " ").trim();
  }

  function parseDenoteEntry(raw) {
    const separator = raw.indexOf("::");
    if (separator === -1) return null;

    const key = htmlText(raw.slice(0, separator));
    const value = raw.slice(separator + 2).trim();
    return key && value ? { key: key, value: value } : null;
  }

  function extractLegacyData(container) {
    const html = container.innerHTML;
    let data = "";
    const cleaned = html.replace(/;;;a([\s\S]*?);;;a/gm, function (_, content) {
      data += "\n" + content;
      return "";
    });
    if (cleaned !== html) container.innerHTML = cleaned;
    return data;
  }

  function extractDenotes(article) {
    const dataBlocks = Array.from(document.querySelectorAll(".omnisyr-denote-data"));
    let raw = "";

    dataBlocks.forEach(function (block) {
      const structuredNotes = Array.from(block.querySelectorAll(":scope > article[lang]"));
      structuredNotes.forEach(function (note) {
        const language = note.lang.toLowerCase().startsWith("zh") ? "zh" : "en";
        if (language !== currentLanguage()) return;

        const term = note.querySelector(":scope > h6");
        const body = note.querySelector(":scope > div");
        const key = term ? term.textContent.replace(/\s+/g, " ").trim() : "";
        if (key && body) denotes.set(key, body.innerHTML);
      });
      if (!structuredNotes.length) raw += "\n" + block.innerHTML;
      block.remove();
    });
    if (denotes.size) return;

    if (!raw) raw = extractLegacyData(article);
    const selected = resolveLegacy(raw);
    selected.replace(/;;;;([\s\S]*?);;;;/gm, function (_, entryText) {
      const entry = parseDenoteEntry(entryText);
      if (entry) denotes.set(entry.key, entry.value);
      return "";
    });
  }

  function headingText(heading) {
    const clone = heading.cloneNode(true);
    clone.querySelectorAll(".omnisyr-heading-anchor").forEach(function (anchor) {
      anchor.remove();
    });
    return clone.textContent.replace(/\s+/g, " ").trim();
  }

  function slugify(value) {
    let slug = value.trim().toLowerCase();
    if (slug.normalize) slug = slug.normalize("NFKC");
    slug = slug
      .replace(/\s+/g, "-")
      .replace(/[^\p{Letter}\p{Number}_-]+/gu, "")
      .replace(/-+/g, "-")
      .replace(/^-|-$/g, "");
    return slug || "section";
  }

  function ensureHeadingIds(headings) {
    const used = new Set();
    headings.forEach(function (heading) {
      let id = heading.id && heading.id.trim();
      if (!id || used.has(id)) {
        const base = slugify(headingText(heading));
        id = base;
        let suffix = 2;
        while (used.has(id) || document.getElementById(id)) {
          id = base + "-" + suffix;
          suffix += 1;
        }
        heading.id = id;
      }
      used.add(id);
    });
  }

  function setupActiveHeading(headings, linksById) {
    if (!("IntersectionObserver" in window)) return;

    const observer = new IntersectionObserver(function (entries) {
      const visible = entries
        .filter(function (entry) { return entry.isIntersecting; })
        .sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
      if (!visible.length) return;

      linksById.forEach(function (link) { link.removeAttribute("aria-current"); });
      const activeLink = linksById.get(visible[0].target.id);
      if (activeLink) activeLink.setAttribute("aria-current", "location");
    }, { rootMargin: "-15% 0px -72% 0px", threshold: 0 });

    headings.forEach(function (heading) { observer.observe(heading); });
  }

  function createTOC(headings) {
    if (!headings.length) return null;

    const details = document.createElement("details");
    details.className = "omnisyr-tool-panel omnisyr-toc";
    details.open = window.matchMedia("(min-width: 1320px)").matches;

    const summary = document.createElement("summary");
    summary.textContent = message("Contents", "目录");
    details.appendChild(summary);

    const navigation = document.createElement("nav");
    navigation.setAttribute("aria-label", message("Table of contents", "文章目录"));
    const linksById = new Map();

    headings.forEach(function (heading) {
      const link = document.createElement("a");
      link.href = "#" + encodeURIComponent(heading.id);
      link.textContent = headingText(heading);
      link.style.setProperty("--omni-toc-level", String(Math.max(0, Number(heading.tagName.slice(1)) - 2)));
      linksById.set(heading.id, link);
      navigation.appendChild(link);
    });

    details.appendChild(navigation);
    setupActiveHeading(headings, linksById);
    return details;
  }

  function renderMathIn(element) {
    if (!element || !window.MathJax || !MathJax.typesetPromise) return Promise.resolve();
    if (MathJax.typesetClear) MathJax.typesetClear([element]);
    return MathJax.typesetPromise([element]).catch(function () {});
  }

  function renderDenote(key, title, content) {
    activeDenote = key;
    title.textContent = key;
    content.setAttribute("aria-busy", "false");

    if (window.MathJax && MathJax.typesetClear) MathJax.typesetClear([content]);
    if (renderedDenotes.has(key)) {
      content.innerHTML = renderedDenotes.get(key);
      return;
    }

    const value = denotes.get(key) || "";
    content.innerHTML = value;
    if (/\$|\\\(|\\\[/.test(value)) {
      content.setAttribute("aria-busy", "true");
      renderMathIn(content).then(function () {
        content.setAttribute("aria-busy", "false");
      });
    }
  }

  function createDenotePanel() {
    if (!denotes.size) return null;

    const panel = document.createElement("section");
    panel.className = "omnisyr-tool-panel omnisyr-denote";
    panel.setAttribute("aria-label", message("Annotation", "注释"));

    const title = document.createElement("div");
    title.className = "omnisyr-denote__title";
    title.textContent = message("Annotation", "注释");

    const content = document.createElement("div");
    content.id = "omnisyr-denote-content";
    content.className = "omnisyr-denote__content";
    content.setAttribute("aria-live", "polite");
    content.textContent = message("Select a highlighted term to read its note.", "选择文中的高亮术语以查看注释。");

    panel.appendChild(title);
    panel.appendChild(content);

    document.querySelectorAll(".notranslate").forEach(function (element) {
      const key = element.textContent.replace(/\s+/g, " ").trim();
      if (!denotes.has(key) || element.hasAttribute("data-omni-denote")) return;

      const label = message("Show annotation", "查看注释") + ": " + key;
      element.setAttribute("data-omni-denote", "ready");
      element.setAttribute("tabindex", "0");
      element.setAttribute("role", "button");
      element.setAttribute("aria-controls", content.id);
      element.setAttribute("aria-label", label);
      element.title = label;
      element.classList.add("omnisyr-denote-trigger");

      const activate = function () { renderDenote(key, title, content); };
      element.addEventListener("click", activate);
      element.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
    });

    return panel;
  }

  function preRenderDenotes(title, content) {
    const mathEntries = Array.from(denotes.entries()).filter(function (entry) {
      return /\$|\\\(|\\\[/.test(entry[1]);
    });
    if (!mathEntries.length) return;

    const cache = document.createElement("div");
    cache.className = "omnisyr-denote-cache";
    const items = mathEntries.map(function (entry) {
      const item = document.createElement("div");
      item.innerHTML = entry[1];
      cache.appendChild(item);
      return { key: entry[0], element: item };
    });
    document.body.appendChild(cache);

    const started = Date.now();
    const finish = function () {
      items.forEach(function (item) {
        renderedDenotes.set(item.key, item.element.innerHTML);
      });
      cache.remove();
      if (activeDenote && renderedDenotes.has(activeDenote)) {
        renderDenote(activeDenote, title, content);
      }
    };

    const attempt = function () {
      if (window.MathJax && MathJax.startup && MathJax.startup.promise && MathJax.typesetPromise) {
        MathJax.startup.promise
          .then(function () { return MathJax.typesetPromise([cache]); })
          .then(finish)
          .catch(function () { cache.remove(); });
        return;
      }
      if (Date.now() - started < 6000) {
        window.setTimeout(attempt, 100);
      } else {
        cache.remove();
      }
    };
    attempt();
  }

  function injectStyle() {
    if (document.getElementById("omnisyr-hyper-toc-style")) return;

    const style = document.createElement("style");
    style.id = "omnisyr-hyper-toc-style";
    style.textContent = `
      .omnisyr-reading-tools {
        display: grid;
        gap: 12px;
        width: 100%;
        margin: 0 0 28px;
      }

      .omnisyr-tool-panel {
        min-width: 0;
        overflow: hidden;
        color: var(--fgColor-default, var(--color-fg-default));
        background: var(--color-canvas-subtle);
        border: 1px solid var(--borderColor-muted, var(--color-border-muted));
        border-radius: var(--omni-radius, 8px);
      }

      .omnisyr-toc > summary {
        display: flex;
        min-height: 44px;
        align-items: center;
        justify-content: space-between;
        padding: 9px 12px;
        font-size: 0.88rem;
        font-weight: 650;
        cursor: pointer;
        list-style: none;
        touch-action: manipulation;
      }

      .omnisyr-toc > summary::-webkit-details-marker {
        display: none;
      }

      .omnisyr-toc > summary::after {
        content: "+";
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 1.1rem;
        font-weight: 400;
      }

      .omnisyr-toc[open] > summary::after {
        content: "−";
      }

      .omnisyr-toc nav {
        max-height: 44vh;
        padding: 4px 8px 9px;
        overflow-y: auto;
        border-top: 1px solid var(--borderColor-muted, var(--color-border-muted));
      }

      .omnisyr-toc nav a {
        display: block;
        padding: 7px 8px 7px calc(8px + var(--omni-toc-level, 0) * 10px);
        overflow-wrap: anywhere;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 0.8rem;
        line-height: 1.35;
        text-decoration: none;
        border-radius: var(--omni-radius-sm, 6px);
      }

      .omnisyr-toc nav a:hover,
      .omnisyr-toc nav a[aria-current="location"] {
        color: var(--omni-accent-fg, var(--color-accent-fg));
        background: var(--omni-accent-soft, var(--color-accent-subtle));
      }

      .omnisyr-denote {
        padding: 12px;
      }

      .omnisyr-denote__title {
        margin-bottom: 8px;
        color: var(--fgColor-default, var(--color-fg-default));
        font-size: 0.88rem;
        font-weight: 650;
      }

      .omnisyr-denote__content {
        max-height: 46vh;
        overflow: auto;
        color: var(--fgColor-muted, var(--color-fg-muted));
        font-size: 0.86rem;
        line-height: 1.6;
        overflow-wrap: anywhere;
        user-select: text;
        -webkit-user-select: text;
      }

      .omnisyr-denote__content > :first-child { margin-top: 0; }
      .omnisyr-denote__content > :last-child { margin-bottom: 0; }

      .omnisyr-denote__content mjx-container[display="true"] {
        display: block !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        overflow-x: auto;
        overflow-y: hidden;
        padding-block: 0.25rem;
      }

      .markdown-body .omnisyr-denote-trigger {
        cursor: help;
        touch-action: manipulation;
      }

      .markdown-body .omnisyr-denote-trigger:focus-visible {
        outline: 2px solid var(--omni-accent);
        outline-offset: 2px;
      }

      .omnisyr-denote-cache {
        position: fixed;
        left: -100000px;
        top: 0;
        width: 520px;
        visibility: hidden;
        pointer-events: none;
        contain: layout paint style;
      }

      @media (min-width: 1320px) {
        .omnisyr-reading-tools {
          position: fixed;
          z-index: 10;
          top: var(--omni-sticky-top, 84px);
          left: calc(50% + 430px);
          width: min(320px, calc(50vw - 450px));
          max-height: calc(100dvh - var(--omni-sticky-top, 84px) - 20px);
          margin: 0;
          overflow-y: auto;
        }
      }

      @media (max-width: 720px) {
        .omnisyr-reading-tools {
          margin-bottom: 22px;
        }

        .omnisyr-toc nav {
          max-height: 36vh;
        }

        .omnisyr-denote__content {
          max-height: 30vh;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function initialize() {
    const contentContainer = document.getElementById("content");
    const article = document.querySelector(".markdown-body");
    if (!contentContainer || !article || document.querySelector(".omnisyr-reading-tools")) return;

    extractDenotes(article);
    const headings = Array.from(article.querySelectorAll("h1, h2, h3, h4, h5, h6"));
    ensureHeadingIds(headings);

    const tools = document.createElement("aside");
    tools.className = "omnisyr-reading-tools";
    tools.setAttribute("aria-label", message("Reading tools", "阅读工具"));

    const toc = createTOC(headings);
    const denotePanel = createDenotePanel();
    if (toc) tools.appendChild(toc);
    if (denotePanel) tools.appendChild(denotePanel);
    if (!tools.children.length) return;

    injectStyle();
    contentContainer.insertBefore(tools, article);

    if (denotePanel) {
      preRenderDenotes(
        denotePanel.querySelector(".omnisyr-denote__title"),
        denotePanel.querySelector(".omnisyr-denote__content")
      );
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
