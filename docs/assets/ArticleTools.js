(function () {
  "use strict";

  function currentLanguage() {
    if (window.OmnisyrLanguage) return window.OmnisyrLanguage.get();
    return document.documentElement.lang.toLowerCase().startsWith("zh") ? "zh" : "en";
  }

  function message(english, chinese) {
    return currentLanguage() === "zh" ? chinese : english;
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

  function uniqueHeadingId(heading, usedIds) {
    const existing = heading.id && heading.id.trim();
    if (existing && !usedIds.has(existing)) {
      usedIds.add(existing);
      return existing;
    }

    const base = slugify(headingText(heading));
    let candidate = base;
    let suffix = 2;
    while (usedIds.has(candidate) || document.getElementById(candidate)) {
      candidate = base + "-" + suffix;
      suffix += 1;
    }
    usedIds.add(candidate);
    heading.id = candidate;
    return candidate;
  }

  function addHeadingAnchors() {
    const article = document.querySelector(".markdown-body");
    if (!article) return;

    const usedIds = new Set();
    article.querySelectorAll("h1, h2, h3, h4").forEach(function (heading) {
      if (heading.querySelector(":scope > .omnisyr-heading-anchor")) return;

      const id = uniqueHeadingId(heading, usedIds);
      const label = message("Link to this section", "链接到本节") + ": " + headingText(heading);
      const anchor = document.createElement("a");
      anchor.className = "omnisyr-heading-anchor";
      anchor.href = "#" + encodeURIComponent(id);
      anchor.title = label;
      anchor.setAttribute("aria-label", label);
      anchor.setAttribute("translate", "no");
      anchor.textContent = "#";
      heading.appendChild(anchor);
    });
  }

  function createImageDialog() {
    if (typeof HTMLDialogElement === "undefined") return null;

    const existing = document.getElementById("omnisyr-image-dialog");
    if (existing) return existing;

    const dialog = document.createElement("dialog");
    dialog.id = "omnisyr-image-dialog";
    dialog.className = "omnisyr-image-dialog";
    dialog.innerHTML =
      '<button class="omnisyr-image-dialog__close" type="button"></button>' +
      '<figure class="omnisyr-image-dialog__figure">' +
        '<img class="omnisyr-image-dialog__image" alt="">' +
        '<figcaption class="omnisyr-image-dialog__caption"></figcaption>' +
      '</figure>';

    const closeButton = dialog.querySelector(".omnisyr-image-dialog__close");
    closeButton.textContent = "×";
    closeButton.title = message("Close image", "关闭图片");
    closeButton.setAttribute("aria-label", closeButton.title);
    closeButton.addEventListener("click", function () { dialog.close(); });

    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) dialog.close();
    });

    document.body.appendChild(dialog);
    return dialog;
  }

  function setupImageZoom() {
    const images = Array.from(document.querySelectorAll(".markdown-body img")).filter(function (image) {
      return !image.closest("a") && !image.classList.contains("emoji");
    });
    if (!images.length) return;

    const dialog = createImageDialog();
    if (!dialog) return;

    const dialogImage = dialog.querySelector(".omnisyr-image-dialog__image");
    const caption = dialog.querySelector(".omnisyr-image-dialog__caption");

    function openImage(image) {
      dialogImage.src = image.currentSrc || image.src;
      dialogImage.alt = image.alt || "";
      caption.textContent = image.alt || "";
      dialog.showModal();
    }

    images.forEach(function (image) {
      if (image.hasAttribute("data-omni-zoom")) return;
      image.setAttribute("data-omni-zoom", "ready");
      image.classList.add("omnisyr-zoomable-image");
      image.setAttribute("tabindex", "0");
      image.setAttribute("role", "button");
      image.setAttribute("aria-label", message("View full-size image", "查看原图") + (image.alt ? ": " + image.alt : ""));

      image.addEventListener("click", function () { openImage(image); });
      image.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openImage(image);
        }
      });
    });
  }

  function setupSameOriginPrefetch() {
    const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    if (connection && (connection.saveData || /(^|-)2g$/.test(connection.effectiveType || ""))) return;

    const prefetched = new Set();
    const pending = new WeakMap();
    const maximumPrefetches = 6;

    function eligibleLink(target) {
      const link = target && target.closest ? target.closest("a[href]") : null;
      if (!link || link.target === "_blank" || link.hasAttribute("download")) return null;

      let url;
      try {
        url = new URL(link.href, location.href);
      } catch (error) {
        return null;
      }
      if (url.origin !== location.origin || !/^https?:$/.test(url.protocol)) return null;
      if (url.pathname === location.pathname || /\.(xml|json|png|jpe?g|gif|svg|webp|pdf)$/i.test(url.pathname)) return null;
      url.hash = "";
      return url.href;
    }

    function prefetch(href) {
      if (!href || prefetched.has(href) || prefetched.size >= maximumPrefetches) return;
      prefetched.add(href);
      const resource = document.createElement("link");
      resource.rel = "prefetch";
      resource.href = href;
      document.head.appendChild(resource);
    }

    document.addEventListener("pointerover", function (event) {
      const link = event.target.closest && event.target.closest("a[href]");
      const href = eligibleLink(event.target);
      if (!link || !href || pending.has(link)) return;
      pending.set(link, window.setTimeout(function () {
        pending.delete(link);
        prefetch(href);
      }, 90));
    }, { passive: true });

    document.addEventListener("pointerout", function (event) {
      const link = event.target.closest && event.target.closest("a[href]");
      if (!link || link.contains(event.relatedTarget)) return;
      const timer = pending.get(link);
      if (timer) window.clearTimeout(timer);
      pending.delete(link);
    }, { passive: true });
  }

  function enhance() {
    addHeadingAnchors();
    setupImageZoom();
    setupSameOriginPrefetch();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhance, { once: true });
  } else {
    enhance();
  }
})();
