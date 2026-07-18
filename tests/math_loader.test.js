const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { containsMathSource } = require("../static/assets/MathLoader.js");

assert.equal(containsMathSource("Plain article text."), false);
assert.equal(containsMathSource("Inline $x_t$ formula."), true);
assert.equal(containsMathSource("Display $$x + y$$ formula."), true);
assert.equal(containsMathSource("\\begin{align}x&=1\\end{align}"), true);

const config = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "config.json"), "utf8"));
assert.ok(
  config.allHead.indexOf("MathLoader.js") < config.allHead.indexOf("HyperTOC.js"),
  "MathLoader must inspect note math before HyperTOC extracts the note data"
);

console.log("math-loader-detection=ok");
