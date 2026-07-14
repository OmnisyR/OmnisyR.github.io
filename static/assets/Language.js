(function () {
  "use strict";

  const STORAGE_KEY = "omnisyr_blog_lang";
  const LANG_ZH = "zh";
  const LANG_EN = "en";
  const META_SEPARATOR = "||";
  const READY_TIMEOUT = 3000;
  const root = document.documentElement;

  const UI_COPY = {
    search: { en: "Search", zh: "搜索" },
    searchSite: { en: "Search posts", zh: "搜索文章" },
    searchPlaceholder: { en: "Search posts…", zh: "搜索文章…" },
    links: { en: "Links", zh: "链接库" },
    home: { en: "Home", zh: "首页" },
    theme: { en: "Switch theme", zh: "切换主题" },
    comments: { en: "Load comments", zh: "加载评论" },
    switchToEnglish: { en: "Switch to English", zh: "切换到英文" },
    switchToChinese: { en: "Switch to Chinese", zh: "切换到中文" }
  };

  function getBrowserLang() {
    const languages = navigator.languages && navigator.languages.length
      ? navigator.languages
      : [navigator.language || navigator.userLanguage || LANG_EN];

    return languages.some(function (language) {
      return String(language).toLowerCase().startsWith("zh");
    }) ? LANG_ZH : LANG_EN;
  }

  function readStoredLang() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored === LANG_ZH || stored === LANG_EN ? stored : null;
    } catch (error) {
      return null;
    }
  }

  function getCurrentLang() {
    return readStoredLang() || getBrowserLang();
  }

  function setCurrentLang(language) {
    try {
      localStorage.setItem(STORAGE_KEY, language);
    } catch (error) {
      // The next load falls back to the browser language in private contexts.
    }
  }

  function copy(key, language) {
    const item = UI_COPY[key];
    return item ? item[language === LANG_ZH ? LANG_ZH : LANG_EN] : key;
  }

  function applyLanguage(language) {
    const normalized = language === LANG_ZH ? LANG_ZH : LANG_EN;
    root.setAttribute("data-omnisyr-lang", normalized);
    root.lang = normalized === LANG_ZH ? "zh-CN" : LANG_EN;
    return normalized;
  }

  function injectCriticalStyle() {
    if (document.getElementById("omnisyr-language-style")) return;

    const style = document.createElement("style");
    style.id = "omnisyr-language-style";
    style.textContent = `
      html[data-omnisyr-i18n="pending"] body {
        visibility: hidden !important;
      }

      html[data-omnisyr-lang="en"] body [lang^="zh"],
      html[data-omnisyr-lang="zh"] body [lang^="en"] {
        display: none !important;
      }

      #omnisyr-language-toggle {
        margin: 0 !important;
      }

      #omnisyr-language-toggle .omnisyr-language-glyph {
        display: block;
        font-family: ui-sans-serif, system-ui, sans-serif;
        font-size: 11px;
        font-weight: 700;
        line-height: 1;
        letter-spacing: 0;
      }
    `;
    document.head.appendChild(style);
  }

  function transLegacy(value, language, forceChinese) {
    if (!value) return value;

    const useChinese = forceChinese || language === LANG_ZH;
    const keepPattern = useChinese
      ? /;;;c([\s\S]*?);;;c/gm
      : /;;;e([\s\S]*?);;;e/gm;
    const removePattern = useChinese
      ? /;;;e([\s\S]*?);;;e/gm
      : /;;;c([\s\S]*?);;;c/gm;

    return value
      .replace(keepPattern, function (_, content) { return content; })
      .replace(removePattern, "");
  }

  function transMeta(value, language, forceChinese) {
    const legacy = transLegacy(value, language, forceChinese);
    if (!legacy || !legacy.includes(META_SEPARATOR)) return legacy;

    const parts = legacy.split(META_SEPARATOR);
    if (parts.length !== 2) return legacy;
    return (forceChinese || language === LANG_ZH ? parts[1] : parts[0]).trim();
  }

  function translateLegacyArticle(language, forceChinese) {
    const article = document.querySelector(".markdown-body");
    if (!article || document.querySelector(".omnisyr-denote-data")) return;

    const html = article.innerHTML;
    if (!/;;;[ec]/.test(html)) return;
    const translated = transLegacy(html, language, forceChinese);
    if (translated !== html) article.innerHTML = translated;
  }

  function translateMetaElement(element, language, forceChinese) {
    if (!element) return;

    const html = element.innerHTML;
    const translated = transMeta(html, language, forceChinese);
    if (translated !== html) element.innerHTML = translated;

    ["title", "aria-label"].forEach(function (attribute) {
      const value = element.getAttribute(attribute);
      if (!value) return;
      element.setAttribute(attribute, transMeta(value, language, forceChinese));
    });
  }

  function translateMetaAttribute(element, attribute, language, forceChinese) {
    if (!element) return;
    const value = element.getAttribute(attribute);
    if (value) element.setAttribute(attribute, transMeta(value, language, forceChinese));
  }

  function setAccessibleLabel(element, value) {
    if (!element) return;
    element.title = value;
    element.setAttribute("aria-label", value);
  }

  function translateChrome(language) {
    const searchButton = document.getElementById("buttonSearch");
    const linksButton = document.querySelector('#header a[href$="/link.html"], #header a[href$="link.html"]');
    const homeButton = document.getElementById("buttonHome");
    const themeButton = document.querySelector('#omnisyr-theme-toggle, #header [onclick*="modeSwitch"]');
    const searchInput = document.querySelector(".subnav-search-input");
    const searchSubmit = document.querySelector(".subnav-search button");
    const commentsButton = document.getElementById("cmButton");

    setAccessibleLabel(searchButton, copy("search", language));
    setAccessibleLabel(linksButton, copy("links", language));
    setAccessibleLabel(homeButton, copy("home", language));
    setAccessibleLabel(themeButton, copy("theme", language));

    if (searchInput) {
      searchInput.setAttribute("aria-label", copy("searchSite", language));
      searchInput.setAttribute("placeholder", copy("searchPlaceholder", language));
      searchInput.setAttribute("name", "q");
      searchInput.setAttribute("autocomplete", "off");
      searchInput.setAttribute("spellcheck", "false");
    }
    if (searchSubmit) searchSubmit.textContent = copy("search", language);
    if (commentsButton) {
      commentsButton.textContent = copy("comments", language);
      setAccessibleLabel(commentsButton, copy("comments", language));
    }

    const runDay = document.getElementById("runday");
    if (runDay) {
      const match = runDay.textContent.match(/\d+/);
      if (match) {
        const days = match[0];
        runDay.textContent = language === LANG_ZH
          ? "网站运行 " + days + " 天 • "
          : "Online for " + days + " days • ";
      }
    }
  }

  function translatePage(language, forceChinese) {
    document.title = transMeta(document.title, language, forceChinese);

    [
      ".blogTitle",
      "#listTitle",
      ".postTitle",
      ".tagTitle",
      "#content > div:first-child:not(.markdown-body):not(#taglabel)"
    ].forEach(function (selector) {
      translateMetaElement(document.querySelector(selector), language, forceChinese);
    });

    [
      "meta[name='description']",
      "meta[property='og:title']",
      "meta[property='og:description']"
    ].forEach(function (selector) {
      translateMetaAttribute(document.querySelector(selector), "content", language, forceChinese);
    });

    document.querySelectorAll(".listTitle, .LabelName, .LabelName a, #taglabel .Label").forEach(function (element) {
      translateMetaElement(element, language, forceChinese);
    });

    translateLegacyArticle(language, forceChinese);
    translateChrome(language);
  }

  function detectForceChinese() {
    const note = document.querySelector(".markdown-alert.markdown-alert-note");
    return !!note && note.textContent.includes("This article currently only supports Chinese.");
  }

  function createLanguageButton(language) {
    if (document.getElementById("omnisyr-language-toggle")) return;

    const titleRight = document.querySelector("#header .title-right, .title-right");
    if (!titleRight) return;

    const button = document.createElement("button");
    button.id = "omnisyr-language-toggle";
    button.className = "btn btn-invisible circle";
    button.type = "button";

    const label = language === LANG_ZH
      ? copy("switchToEnglish", language)
      : copy("switchToChinese", language);
    setAccessibleLabel(button, label);
    button.innerHTML = '<span class="omnisyr-language-glyph" aria-hidden="true" translate="no">文/A</span>';

    button.addEventListener("click", function () {
      setCurrentLang(language === LANG_ZH ? LANG_EN : LANG_ZH);
      location.reload();
    });

    const themeButton = Array.from(titleRight.children).find(function (item) {
      return item.id === "omnisyr-theme-toggle" || String(item.getAttribute("onclick") || "").includes("modeSwitch");
    });
    titleRight.insertBefore(button, themeButton || null);
  }

  function observeTagPage(language, forceChinese) {
    const target = document.getElementById("content");
    if (!target || !document.querySelector(".tagTitle") || !window.MutationObserver) return;

    let scheduled = false;
    const observer = new MutationObserver(function () {
      if (scheduled) return;
      scheduled = true;
      window.setTimeout(function () {
        scheduled = false;
        document.querySelectorAll(".listTitle, #taglabel .Label").forEach(function (element) {
          translateMetaElement(element, language, forceChinese);
        });
        translateChrome(language);
      }, 16);
    });
    observer.observe(target, { childList: true, subtree: true });
  }

  const initialLanguage = applyLanguage(getCurrentLang());
  root.setAttribute("data-omnisyr-i18n", "pending");
  injectCriticalStyle();

  const readyFailsafe = window.setTimeout(function () {
    if (document.body) translatePage(initialLanguage, false);
    root.setAttribute("data-omnisyr-i18n", "ready");
  }, READY_TIMEOUT);

  window.OmnisyrLanguage = {
    get: function () {
      return root.getAttribute("data-omnisyr-lang") === LANG_ZH ? LANG_ZH : LANG_EN;
    },
    text: function (key) {
      return copy(key, this.get());
    },
    resolveLegacy: function (value) {
      return transLegacy(value, this.get(), false);
    },
    resolveMeta: function (value) {
      return transMeta(value, this.get(), false);
    }
  };

  document.addEventListener("DOMContentLoaded", function () {
    const forceChinese = detectForceChinese();
    const language = applyLanguage(forceChinese ? LANG_ZH : initialLanguage);

    translatePage(language, forceChinese);
    observeTagPage(language, forceChinese);
    if (!forceChinese) createLanguageButton(language);

    window.clearTimeout(readyFailsafe);
    root.setAttribute("data-omnisyr-i18n", "ready");
    document.dispatchEvent(new CustomEvent("omnisyr:language-ready", {
      detail: { language: language, forceChinese: forceChinese }
    }));
  });
})();
