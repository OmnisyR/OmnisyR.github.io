const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {
  containsMathSource,
  shouldTypesetBlock
} = require("../static/assets/MathLoader.js");
const { languageMatches } = require("../static/assets/HyperTOC.js");

assert.equal(containsMathSource("Plain article text."), false);
assert.equal(containsMathSource("Inline $x_t$ formula."), true);
assert.equal(containsMathSource("Display $$x + y$$ formula."), true);
assert.equal(containsMathSource("\\begin{align}x&=1\\end{align}"), true);

assert.equal(shouldTypesetBlock(null, false, "en"), true);
assert.equal(shouldTypesetBlock("en", false, "en"), true);
assert.equal(shouldTypesetBlock("zh-CN", false, "en"), false);
assert.equal(shouldTypesetBlock("zh-CN", false, "zh"), true);
assert.equal(shouldTypesetBlock(null, true, "en"), false);

assert.equal(languageMatches("en", "en"), true);
assert.equal(languageMatches("en-US", "en"), true);
assert.equal(languageMatches("zh-CN", "zh"), true);
assert.equal(languageMatches("zh-CN", "en"), false);

const config = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "config.json"), "utf8"));
assert.ok(
  config.allHead.indexOf("MathLoader.js") < config.allHead.indexOf("HyperTOC.js"),
  "MathLoader must inspect note math before HyperTOC extracts the note data"
);

const mathLoader = fs.readFileSync(
  path.join(__dirname, "..", "static", "assets", "MathLoader.js"),
  "utf8"
);
const language = fs.readFileSync(
  path.join(__dirname, "..", "static", "assets", "Language.js"),
  "utf8"
);
const hyperToc = fs.readFileSync(
  path.join(__dirname, "..", "static", "assets", "HyperTOC.js"),
  "utf8"
);

assert.doesNotMatch(
  mathLoader,
  /data-omnisyr-math=["']pending["'][\s\S]*?visibility:\s*hidden/,
  "MathJax loading must not hide the article"
);
assert.doesNotMatch(
  language,
  /data-omnisyr-i18n=["']pending["'][\s\S]*?visibility:\s*hidden/,
  "language initialization must not hide the whole page"
);
assert.match(mathLoader, /displayOverflow:\s*["']linebreak["']/);
assert.match(
  mathLoader,
  /typesetPromise\(typesetTargets\(article\)\)/,
  "shared formulas outside language wrappers must also be typeset"
);
assert.match(hyperToc, /filter\(headingMatchesCurrentLanguage\)/);

const denoteContentRule = hyperToc.match(/\.omnisyr-denote__content\s*\{([\s\S]*?)\}/);
assert.ok(denoteContentRule, "annotation content styles must exist");
assert.doesNotMatch(denoteContentRule[1], /max-height/);
assert.doesNotMatch(denoteContentRule[1], /overflow:\s*auto/);

console.log("blog-ui-regressions=ok");
