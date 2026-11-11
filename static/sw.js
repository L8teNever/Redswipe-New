// Sparsam: kein Push, kein Background-Sync, kein Periodic-Sync. Ist die App zu, läuft hier nichts.
// Einzige Aufgabe außer Installierbarkeit: die Seite selbst (nur Seitenaufrufe) für den Offline-Start merken,
// damit gespeicherte Offline-Videos auch ohne Netz abspielbar sind. Alle anderen Anfragen gehen unverändert ans Netz.
const SHELL = 'ph-shell-v1';
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (e) => {
  const r = e.request;
  if (r.mode !== 'navigate' || r.method !== 'GET') return;
  const p = new URL(r.url).pathname;
  if (!['/', '/pimpbunny', '/abos', '/favoriten', '/suche', '/shorts', '/offline'].includes(p.replace(/\/+$/, '') || '/')) return;
  e.respondWith(fetch(r).then((res) => {
    if (res.ok && res.type === 'basic') { const c = res.clone(); caches.open(SHELL).then((s) => s.put('/', c)).catch(() => {}); }
    return res;
  }).catch(() => caches.open(SHELL).then((s) => s.match('/')).then((m) => m || Response.error())));
});
