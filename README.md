# Reddit Swipe

Eine selbst gehostete Web-App, mit der du Bilder und Videos aus Subreddits im Vollbild durchwischen kannst. Der Server holt nur öffentliche Reddit-Daten und reicht Medien über einen Proxy aus.

## Start mit Docker

```bash
docker compose up -d
```

Danach im Browser: http://localhost:8080

Die App sucht im Feld nach einem Subreddit-Link, z. B. `https://www.reddit.com/r/aww/`.

## Umgebungsvariablen

| Variable | Standard | Bedeutung |
|---|---|---|
| `PORT` | `8080` | Port des Servers |
| `STATIC_DIR` | `static` | Ordner mit der Web-Oberfläche (im Image `/app/static`) |
| `USER_AGENT` | `reddit-swipe/0.1 …` | User-Agent für Reddit-Anfragen |
| `HTTPS_PROXY` | – | Optionaler Proxy für ausgehende Anfragen |

## Hinweis zum ghcr-Paket

Nach dem ersten Build ist das Paket auf ghcr.io evtl. **privat**. Auf GitHub unter *Packages → reddit-swipe → Package settings → Change visibility → Public* stellen (oder mit `docker login ghcr.io` anmelden).

## Lokal ohne Docker

```bash
cargo run --release
```

Dann ebenfalls http://localhost:8080 öffnen (Rust-Toolchain nötig).

## Hinweis zu Reddit

Reddit blockt manchmal Server-IPs (HTTP 403 oder 429). Das ist kein Fehler im Code. Dann hilft `HTTPS_PROXY` oder ein Betrieb auf einem Heim-Rechner.

Es werden nur öffentlich zugängliche Reddit-Daten abgerufen.

## VPN für Reddit (optional)

Reddit blockt oft Rechenzentrums-IPs (HTTP 403). Dann kann der Verkehr über einen Heim-Anschluss laufen:
Im Dialog **VPN** (oben in der Seite) lädst du eine WireGuard-`.conf` hoch (z. B. aus einer Fritz!Box). Die App prüft
Format und Keys (je 44 Zeichen, Endung `=`), schreibt eine normalisierte Config nach `VPN_DIR` (Standard `/vpn`)
und zeigt die ausgehende IP an. Ein `gluetun`-Container liest die Datei und stellt einen HTTP-Proxy bereit;
die App nutzt ihn über `HTTPS_PROXY`. Der Private Key wird nie an den Browser zurückgegeben.

- Der Upload ist nur mit Cloudflare-Access-Header erlaubt (`REQUIRE_ACCESS=0` schaltet das für lokale Tests ab).
- Hostnamen im Endpoint (z. B. `xyz.myfritz.net`) werden alle 5 Minuten neu aufgelöst.
- Ein Beispiel-Stack mit `gluetun` und Wächter steht in `docker-compose.vpn.example.yml`.

## NudeNet-Scan (optional)

Der Ordner `nsfw/` ist ein eigener Container (Python, NudeNet, ffmpeg). Er teilt sich das Cache-Volume (`/cache`) mit dem Backend
und wird über `NSFW_URL=http://<name>:8000` angesprochen. Sobald ein Feed geladen ist, scannt das Backend alle Bilder und
Videos im Hintergrund (Videos Frame für Frame, zusammengefasst pro Label). Die Ergebnisse liegen als `*.scan.json` im Cache.
Im Frontend zeigt der Knopf „Anzeigen“ sie an (Boxen bei Bildern, Zusammenfassung bei Videos). Ohne `NSFW_URL` ist alles aus.
