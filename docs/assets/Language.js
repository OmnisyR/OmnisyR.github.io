(function () {
  const STORAGE_KEY = "omnisyr_blog_lang";
  const LANG_CN = "zh";
  const LANG_EN = "en";

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

  function translate(container, lang, forceCn) {
    if (!container) return;
    container.innerHTML = transStr(container.innerHTML, lang, forceCn);
  }

  function detectForceCn() {
    const elements = document.getElementsByClassName("markdown-alert markdown-alert-note");
    return elements.length > 0 &&
      elements[0].innerHTML.includes("This article currently only supports Chinese.");
  }

  function createLanguageButton(lang) {
    const button = document.createElement("button");
    button.id = "omnisyr-language-toggle";
    button.type = "button";
    button.title = lang === LANG_CN ? "切换到英文" : "Switch to Chinese";
    button.setAttribute("aria-label", button.title);

    button.innerHTML = `
      <svg class="omnisyr-language-icon" viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="9"></circle>
        <path d="M3 12h18"></path>
        <path d="M12 3c2.2 2.5 3.4 5.5 3.4 9s-1.2 6.5-3.4 9"></path>
        <path d="M12 3c-2.2 2.5-3.4 5.5-3.4 9s1.2 6.5 3.4 9"></path>
      </svg>
      <span>${lang === LANG_CN ? "EN" : "中"}</span>
    `;

    button.addEventListener("click", function () {
      const nextLang = getCurrentLang() === LANG_CN ? LANG_EN : LANG_CN;
      setCurrentLang(nextLang);
      location.reload();
    });

    document.body.appendChild(button);
  }

  function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
      #omnisyr-language-toggle {
        position: fixed;
        right: 18px;
        bottom: 18px;
        z-index: 9999;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        height: 36px;
        padding: 0 11px;
        border: 1px solid var(--borderColor-default, #d0d7de);
        border-radius: 999px;
        background: var(--bgColor-default, #ffffff);
        color: var(--fgColor-default, #24292f);
        box-shadow: 0 6px 18px rgba(27, 31, 36, 0.12);
        cursor: pointer;
        font-size: 14px;
        line-height: 1;
      }

      #omnisyr-language-toggle:hover {
        background: var(--bgColor-muted, #f6f8fa);
      }

      .omnisyr-language-icon {
        width: 17px;
        height: 17px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.8;
        stroke-linecap: round;
        stroke-linejoin: round;
      }
    `;
    document.head.appendChild(style);
  }

  document.addEventListener("DOMContentLoaded", function () {
    const lang = getCurrentLang();
    const forceCn = detectForceCn();

    document.documentElement.lang = forceCn || lang === LANG_CN ? "zh-CN" : "en";
    document.title = transStr(document.title, lang, forceCn);

    translate(document.getElementById("listTitle"), lang, forceCn);
    translate(document.getElementById("content"), lang, forceCn);

    let elements = document.getElementsByClassName("postTitle");
    if (elements.length > 0) {
      translate(elements[0], lang, forceCn);
    }

    elements = document.getElementsByClassName("Label LabelName");
    for (const item of elements) {
      translate(item, lang, forceCn);
    }

    injectStyle();

    if (!forceCn) {
      createLanguageButton(lang);
    }
  });
})();
