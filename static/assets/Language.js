(function () {
  const STORAGE_KEY = "omnisyr_blog_lang";
  const LANG_CN = "zh";
  const LANG_EN = "en";
  const META_SEPARATOR = "||";

  function getBrowserLang() {
    const langs = navigator.languages && navigator.languages.length
      ? navigator.languages
      : [navigator.language || navigator.userLanguage || "en"];

    return langs.some(lang => String(lang).toLowerCase().startsWith("zh"))
      ? LANG_CN
      : LANG_EN;
  }

  function getCurrentLang() {
    return localStorage.getItem(STORAGE_KEY) || getBrowserLang();
  }

  function setCurrentLang(lang) {
    localStorage.setItem(STORAGE_KEY, lang);
  }

  function transStr(str, lang, forceCn) {
    if (!str) return str;

    const useCn = forceCn || lang === LANG_CN;

    const keepPattern = useCn
      ? /;;;c([\s\S]*?);;;c/gm
      : /;;;e([\s\S]*?);;;e/gm;

    const removePattern = useCn
      ? /;;;e([\s\S]*?);;;e/gm
      : /;;;c([\s\S]*?);;;c/gm;

    return str
      .replace(keepPattern, function (_, content) {
        return content;
      })
      .replace(removePattern, "");
  }

  function transMeta(str, lang, forceCn) {
    const legacy = transStr(str, lang, forceCn);
    if (!legacy.includes(META_SEPARATOR)) return legacy;

    const parts = legacy.split(META_SEPARATOR);
    if (parts.length !== 2) return legacy;

    return (forceCn || lang === LANG_CN ? parts[1] : parts[0]).trim();
  }

  function translateLegacy(container, lang, forceCn) {
    if (!container) return;
    container.innerHTML = transStr(container.innerHTML, lang, forceCn);
  }

  function translateMetaElement(element, lang, forceCn) {
    if (!element) return;
    element.innerHTML = transMeta(element.innerHTML, lang, forceCn);
    if (element.title) element.title = transMeta(element.title, lang, forceCn);
    if (element.getAttribute("aria-label")) {
      element.setAttribute("aria-label", transMeta(element.getAttribute("aria-label"), lang, forceCn));
    }
  }

  function translateMetaAttribute(element, attr, lang, forceCn) {
    if (!element || !element.getAttribute(attr)) return;
    element.setAttribute(attr, transMeta(element.getAttribute(attr), lang, forceCn));
  }

  function applyLanguageVisibility(lang, forceCn) {
    const visibleLang = forceCn || lang === LANG_CN ? LANG_CN : LANG_EN;
    document.documentElement.setAttribute("data-omnisyr-lang", visibleLang);
    document.documentElement.lang = visibleLang === LANG_CN ? "zh-CN" : "en";
  }

  function detectForceCn() {
    const elements = document.getElementsByClassName("markdown-alert markdown-alert-note");
    return elements.length > 0 &&
      elements[0].innerHTML.includes("This article currently only supports Chinese.");
  }

  function createLanguageButton(lang) {
    const button = document.createElement("a");
    button.id = "omnisyr-language-toggle";
    button.className = "btn btn-invisible circle";
    button.href = "javascript:void(0)";
  
    button.title = lang === LANG_CN ? "Switch to English" : "切换到中文";
    button.setAttribute("aria-label", button.title);
    button.setAttribute("role", "button");
  
    button.innerHTML = `<svg class="octicon omnisyr-language-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M 3 8 h 10 M 8 5 v 3 M 5 8 c 1 3 2 5 7 7 M 11 8 c -1 3 -2 5 -7 7 M 14 20 l 4 -11 l 4 11 M 15 17 h 6"></path></svg>`;
  
   button.addEventListener("click", function (event) {
      event.preventDefault();
  
      const nextLang = getCurrentLang() === LANG_CN ? LANG_EN : LANG_CN;
      setCurrentLang(nextLang);
      location.reload();
    });
  
    const titleRight =
      document.querySelector("#header .title-right") ||
      document.querySelector(".title-right");
  
    if (titleRight) {
      const themeButton = Array.from(titleRight.children).find(function (item) {
        return item.getAttribute("onclick") === "modeSwitch()";
      });
  
      titleRight.insertBefore(button, themeButton || null);
    }
  }
  
  function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
      html[data-omnisyr-lang="en"] body [lang^="zh"],
      html[data-omnisyr-lang="zh"] body [lang^="en"] {
        display: none !important;
      }

      #omnisyr-language-toggle {
        margin: 0 !important;
      }
  
      #omnisyr-language-toggle .omnisyr-language-icon {
        width: 16px;
        height: 16px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.65;
        stroke-linecap: round;
        stroke-linejoin: round;
        display: block;
      }
    `;
    document.head.appendChild(style);
  }

  function translatePage(lang, forceCn) {
    document.title = transMeta(document.title, lang, forceCn);

    translateMetaElement(document.querySelector(".blogTitle"), lang, forceCn);
    translateMetaElement(document.getElementById("listTitle"), lang, forceCn);
    translateMetaElement(document.querySelector(".postTitle"), lang, forceCn);
    translateMetaElement(document.querySelector(".tagTitle"), lang, forceCn);
    translateMetaElement(document.querySelector("#content > div:first-child:not(.markdown-body):not(#taglabel)"), lang, forceCn);

    [
      "meta[name='description']",
      "meta[property='og:title']",
      "meta[property='og:description']"
    ].forEach(function (selector) {
      translateMetaAttribute(document.querySelector(selector), "content", lang, forceCn);
    });

    const content = document.getElementById("content");
    if (content) translateLegacy(content, lang, forceCn);

    const metaSelectors = [
      ".listTitle",
      ".LabelName",
      ".LabelName a",
      "#taglabel .Label",
      "#footer",
      "[title]",
      "[aria-label]"
    ];

    metaSelectors.forEach(function (selector) {
      document.querySelectorAll(selector).forEach(function (element) {
        translateMetaElement(element, lang, forceCn);
      });
    });
  }

  function observeDynamicContent(lang, forceCn) {
    const target = document.getElementById("content");
    if (!target || !window.MutationObserver) return;

    let scheduled = false;
    const observer = new MutationObserver(function () {
      if (scheduled) return;
      scheduled = true;
      window.requestAnimationFrame(function () {
        scheduled = false;
        translatePage(lang, forceCn);
      });
    });

    observer.observe(target, { childList: true, subtree: true });
  }

  document.addEventListener("DOMContentLoaded", function () {
    const lang = getCurrentLang();
    const forceCn = detectForceCn();

    applyLanguageVisibility(lang, forceCn);
    injectStyle();
    translatePage(lang, forceCn);
    observeDynamicContent(lang, forceCn);

    if (!forceCn) {
      createLanguageButton(lang);
    }
  });
})();
