// Bewusst minimal: nur damit die Seite als App installierbar ist.
// Kein fetch-Handler (jede Anfrage geht direkt ans Netz, der Worker wird dafür nicht geweckt),
// kein Push, kein Background-Sync, kein Periodic-Sync, kein Cache. Ist die App zu, läuft hier nichts.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
