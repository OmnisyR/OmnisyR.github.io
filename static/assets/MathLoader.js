(function (root) {
  "use strict";

  const MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@4.1.3/tex-chtml.js";
  const LOAD_TIMEOUT = 12000;

  function containsMathSource(value) {
    const text = String(value || "");
    return /\\begin\{(?:align|align\*|equation|equation\*|gather|gather\*|multline|multline\*)\}/.test(text)
      || /\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]/.test(text)
      || /\$\$[\s\S]+?\$\$/.test(text)
      || /(^|[^\\$])\$(?!\s|\d)(?:\\.|[^$\n])+?\$(?!\$)/m.test(text);
  }

  function normalizeLanguage(value) {
    return String(value || "").toLowerCase().startsWith("zh") ? "zh" : "en";
  }

  function shouldTypesetBlock(blockLanguage, isNoteData, selectedLanguage) {
    if (isNoteData) return false;
    if (!blockLanguage) return true;
    return normalizeLanguage(blockLanguage) === normalizeLanguage(selectedLanguage);
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      containsMathSource: containsMathSource,
      shouldTypesetBlock: shouldTypesetBlock
    };
    return;
  }

  const document = root.document;
  if (!document) return;

  function articleMathSource(article) {
    const clone = article.cloneNode(true);
    clone.querySelectorAll("pre, code, script, style, textarea").forEach(function (element) {
      element.remove();
    });
    return clone.textContent || "";
  }

  function addPreconnect() {
    if (document.querySelector('link[data-omnisyr-math-preconnect]')) return;
    const link = document.createElement("link");
    link.rel = "preconnect";
    link.href = "https://cdn.jsdelivr.net";
    link.crossOrigin = "anonymous";
    link.setAttribute("data-omnisyr-math-preconnect", "ready");
    document.head.appendChild(link);
  }

  function typesetTargets(article) {
    const selectedLanguage = document.documentElement.getAttribute("data-omnisyr-lang") || "en";
    const targets = Array.from(article.children).filter(function (element) {
      return shouldTypesetBlock(
        element.getAttribute("lang"),
        element.classList.contains("omnisyr-denote-data"),
        selectedLanguage
      );
    });
    return targets.length ? targets : [article];
  }

  function initialize() {
    const article = document.querySelector(".markdown-body");
    if (!article || !containsMathSource(articleMathSource(article))) {
      document.documentElement.setAttribute("data-omnisyr-math", "ready");
      return;
    }

    addPreconnect();
    document.documentElement.setAttribute("data-omnisyr-math", "pending");
    article.setAttribute("aria-busy", "true");

    let settleReady;
    const ready = new Promise(function (resolve) { settleReady = resolve; });
    let settled = false;
    let timeoutId;

    function finish(state, error) {
      if (settled) return;
      settled = true;
      root.clearTimeout(timeoutId);
      article.removeAttribute("aria-busy");
      document.documentElement.setAttribute("data-omnisyr-math", state);
      settleReady({ state: state, error: error || null });
      document.dispatchEvent(new CustomEvent("omnisyr:math-ready", {
        detail: { state: state, error: error || null }
      }));
    }

    root.OmnisyrMath = {
      ready: ready,
      typeset: function (elements) {
        return ready.then(function () {
          if (!root.MathJax || !root.MathJax.typesetPromise) return;
          if (root.MathJax.typesetClear) root.MathJax.typesetClear(elements);
          return root.MathJax.typesetPromise(elements);
        });
      }
    };

    root.MathJax = {
      tex: {
        inlineMath: { "[+]": [["$", "$"]] },
        processEscapes: true
      },
      startup: {
        typeset: false
      },
      output: {
        displayOverflow: "linebreak",
        linebreaks: {
          inline: true,
          width: "100%",
          lineleading: 0.2
        }
      }
    };

    const script = document.createElement("script");
    script.id = "omnisyr-mathjax";
    script.src = MATHJAX_URL;
    script.async = true;
    script.crossOrigin = "anonymous";
    script.referrerPolicy = "strict-origin-when-cross-origin";
    script.fetchPriority = "high";

    script.addEventListener("load", function () {
      const startup = root.MathJax && root.MathJax.startup && root.MathJax.startup.promise
        ? root.MathJax.startup.promise
        : Promise.resolve();
      startup
        .then(function () { return root.MathJax.typesetPromise(typesetTargets(article)); })
        .then(function () { finish("ready"); })
        .catch(function (error) {
          console.error("MathJax typesetting failed", error);
          finish("error", String(error));
        });
    });
    script.addEventListener("error", function () {
      finish("error", "MathJax failed to load");
    });

    timeoutId = root.setTimeout(function () {
      finish("error", "MathJax loading timed out");
    }, LOAD_TIMEOUT);
    document.head.appendChild(script);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})(typeof window !== "undefined" ? window : globalThis);
