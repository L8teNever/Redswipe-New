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
