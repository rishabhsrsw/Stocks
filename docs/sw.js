// App-shell cache so the home-screen app opens instantly and offline.
// Picks data is cached by the page itself (see loadData in index.html).
const SHELL = "shell-v1";
const FILES = ["./", "index.html", "manifest.webmanifest", "icons/icon-192.png", "icons/apple-touch-icon.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k.startsWith("shell-") && k !== SHELL).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.includes("/data/")) return;
  // Network first (so updates and sign-in redirects work), cached copy when offline.
  e.respondWith(
    fetch(e.request)
      .then(r => {
        if (r.ok && r.type === "basic") caches.open(SHELL).then(c => c.put(e.request, r.clone()));
        return r;
      })
      .catch(() => caches.match(e.request).then(r => r || caches.match("index.html")))
  );
});
