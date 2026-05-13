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
    button.className = "btn btn-invisible circle";
  
    button.title = lang === LANG_CN ? "Switch to English" : "切换到中文";
    button.setAttribute("aria-label", button.title);
  
    button.innerHTML = `<svg class="octicon omnisyr-language-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M 3 8 h 10 M 8 5 v 3 M 5 8 c 1 3 2 5 7 7 M 11 8 c -1 3 -2 5 -7 7 M 14 20 l 4 -11 l 4 11 M 15 17 h 6"></path></svg>`;
  
    button.addEventListener("click", function () {
      const nextLang = getCurrentLang() === LANG_CN ? LANG_EN : LANG_CN;
      setCurrentLang(nextLang);
      location.reload();
    });
  
    const titleRight = document.querySelector("#header .title-right") || document.querySelector(".title-right");
  
    if (titleRight) {
      const themeButton = Array.from(titleRight.children).find(function (item) {
        return item.getAttribute("onclick") === "modeSwitch()";
      });
  
      titleRight.insertBefore(button, themeButton || null);
    } else {
      document.body.appendChild(button);
    }
  }
  
  function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
      #omnisyr-language-toggle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--fgColor-muted, #57606a);
        cursor: pointer;
        line-height: 1;
        vertical-align: middle;
      }
  
      #omnisyr-language-toggle:hover {
        color: var(--fgColor-accent, #0969da);
      }
  
      .omnisyr-language-icon {
        width: 16px;
        height: 16px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.65;
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
