// Windows app: shows the shared dashboard and reads the weekly picks straight from the
// private GitHub repo with a read-only token, so no website or hosting is needed.
const { app, BrowserWindow, ipcMain, Menu, safeStorage, shell } = require("electron");
const fs = require("fs");
const path = require("path");

const DEFAULT_REPO = "rishabhsrsw/Stocks";
const DATA_FILES = ["latest", "daily", "stocks", "us/latest", "us/daily", "us/stocks"];  // docs/data/<name>.json
const userFile = (name) => path.join(app.getPath("userData"), name);

let mainWin = null;
let settingsWin = null;

function readSettings() {
  let repo = DEFAULT_REPO;
  let token = "";
  try { repo = JSON.parse(fs.readFileSync(userFile("settings.json"), "utf8")).repo || repo; } catch (_) {}
  try {
    const raw = fs.readFileSync(userFile("token.bin"));
    token = safeStorage.isEncryptionAvailable() ? safeStorage.decryptString(raw) : raw.toString("utf8");
  } catch (_) {}
  return { repo, token };
}

function writeSettings({ repo, token }) {
  fs.mkdirSync(app.getPath("userData"), { recursive: true });
  fs.writeFileSync(userFile("settings.json"), JSON.stringify({ repo }));
  // Encrypted with the Windows user account (DPAPI); never stored in plain text on Windows.
  const data = safeStorage.isEncryptionAvailable() ? safeStorage.encryptString(token) : Buffer.from(token, "utf8");
  fs.writeFileSync(userFile("token.bin"), data);
}

async function fetchPicks(repo, token, name = "latest") {
  const res = await fetch(`https://api.github.com/repos/${repo}/contents/docs/data/${name}.json`, {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github.raw+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "nse-weekly-picks-desktop",
    },
  });
  if (res.status === 401) throw new Error("GitHub rejected the token (expired or wrong). Open Settings to update it.");
  if (res.status === 404) throw new Error(`No picks found in ${repo}. Check the repo name and token access, and that the Weekly picks Action has run.`);
  if (!res.ok) throw new Error(`GitHub returned HTTP ${res.status}`);
  return res.json();
}

ipcMain.handle("picks:load", async (_e, name = "latest") => {
  if (!DATA_FILES.includes(name)) throw new Error("Unknown data file");
  if (process.env.NSE_PICKS_DIR) {  // development: preview the app with local docs/data files
    return { data: JSON.parse(fs.readFileSync(path.join(process.env.NSE_PICKS_DIR, `${name}.json`), "utf8")), stale: false };
  }
  const { repo, token } = readSettings();
  if (!token) throw new Error("NO_TOKEN");
  const cache = userFile(`${name.replace("/", "-")}.json`);  // "us/latest" -> us-latest.json
  try {
    const data = await fetchPicks(repo, token, name);
    fs.writeFileSync(cache, JSON.stringify(data));
    return { data, stale: false };
  } catch (err) {
    // Offline: fall back to the last copy saved on this PC.
    if (err instanceof TypeError && fs.existsSync(cache)) {
      return { data: JSON.parse(fs.readFileSync(cache, "utf8")), stale: true };
    }
    throw err;
  }
});

// Live prices for the held tickers, fetched here because a web page can't call Yahoo directly.
ipcMain.handle("quotes:fetch", async (_e, tickers) => {
  const out = {};
  await Promise.all((Array.isArray(tickers) ? tickers : []).filter((t) => /^[\w.^&-]{1,20}$/.test(t)).slice(0, 100).map(async (t) => {
    try {
      const res = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(t)}?range=1d&interval=1d`,
        { headers: { "User-Agent": "Mozilla/5.0" }, signal: AbortSignal.timeout(8000) });
      if (!res.ok) return;
      const m = (await res.json()).chart.result[0].meta;
      out[t] = { price: m.regularMarketPrice, time: m.regularMarketTime };
    } catch (_) {}
  }));
  return out;
});

ipcMain.handle("settings:get", () => {
  const { repo, token } = readSettings();
  return { repo, hasToken: Boolean(token) };
});

ipcMain.handle("settings:save", async (_e, { repo, token }) => {
  repo = (repo || DEFAULT_REPO).trim();
  token = (token || "").trim() || readSettings().token;
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) return { ok: false, error: "Repository must look like owner/name." };
  if (!token) return { ok: false, error: "Paste your GitHub token." };
  try {
    await fetchPicks(repo, token);  // validate before saving
  } catch (err) {
    return { ok: false, error: err.message };
  }
  writeSettings({ repo, token });
  if (settingsWin) settingsWin.close();
  if (mainWin) mainWin.reload();
  return { ok: true };
});

ipcMain.handle("settings:open", () => openSettings());

const secureWebPrefs = {
  preload: path.join(__dirname, "preload.js"),
  contextIsolation: true,
  nodeIntegration: false,
  sandbox: true,
};

function openExternalLinks(win) {
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith("file://")) { e.preventDefault(); if (url.startsWith("https://")) shell.openExternal(url); }
  });
}

function openSettings() {
  if (settingsWin) { settingsWin.focus(); return; }
  settingsWin = new BrowserWindow({
    width: 520, height: 560, parent: mainWin, modal: true, resizable: false,
    title: "Settings", autoHideMenuBar: true, webPreferences: secureWebPrefs,
  });
  openExternalLinks(settingsWin);
  settingsWin.loadFile(path.join(__dirname, "settings.html"));
  settingsWin.on("closed", () => { settingsWin = null; });
}

function createWindow() {
  mainWin = new BrowserWindow({
    width: 1240, height: 900, minWidth: 380, title: "NSE Weekly Picks",
    icon: path.join(__dirname, "web", "icons", "icon-512.png"),
    backgroundColor: "#0d0d0d", autoHideMenuBar: true, webPreferences: secureWebPrefs,
  });
  openExternalLinks(mainWin);
  mainWin.loadFile(path.join(__dirname, "web", "index.html"));
  if (!readSettings().token) mainWin.webContents.once("did-finish-load", openSettings);
}

Menu.setApplicationMenu(Menu.buildFromTemplate([
  { label: "App", submenu: [
    { label: "Refresh", accelerator: "F5", click: () => mainWin && mainWin.reload() },
    { label: "Settings…", accelerator: "CmdOrCtrl+,", click: openSettings },
    { type: "separator" },
    { role: "quit" },
  ] },
  { label: "View", submenu: [{ role: "zoomIn" }, { role: "zoomOut" }, { role: "resetZoom" }, { role: "togglefullscreen" }] },
]));

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => { if (mainWin) { if (mainWin.isMinimized()) mainWin.restore(); mainWin.focus(); } });
  app.whenReady().then(createWindow);
  app.on("window-all-closed", () => app.quit());
}
