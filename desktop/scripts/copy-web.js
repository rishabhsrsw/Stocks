// Copies the shared dashboard (docs/index.html + icons) into the app, so the website,
// iPhone home-screen app and Windows app always render the same screen.
const fs = require("fs");
const path = require("path");

const src = path.join(__dirname, "..", "..", "docs");
const dst = path.join(__dirname, "..", "web");
fs.rmSync(dst, { recursive: true, force: true });
fs.mkdirSync(path.join(dst, "icons"), { recursive: true });
fs.copyFileSync(path.join(src, "index.html"), path.join(dst, "index.html"));
for (const f of fs.readdirSync(path.join(src, "icons"))) {
  fs.copyFileSync(path.join(src, "icons", f), path.join(dst, "icons", f));
}
console.log("Copied dashboard into", dst);
