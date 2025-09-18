# Agents.md

## TL;DR

InfuseAtmosAdder is a tiny reverse-proxy for **Infuse → Jellyfin** that can **live-remux** a movie with a **filename-based sidecar audio track** (e.g. `Movie.en.ec3` / `Movie.eng.ac3`) via `ffmpeg` **copy mode**—no permanent remux.
When the client is Infuse and `MUX_ON=true`, the proxy looks for a sidecar next to the video file and streams **MKV** (or `MP4` if explicitly configured). Otherwise, it **passes through** the Jellyfin stream unchanged.

Key invariants you must **not break**:

* Sidecar detection is **filesystem-only** (no Jellyfin DeliveryUrl/ExternalStream logic).
* Infuse detection by **User-Agent** (“infuse” substring).
* Keep **direct play** behaviour identical for non-Infuse or when muxing is disabled.
* The proxy **preserves Jellyfin auth headers**, it does not invent new auth.
* Default container is **MKV** (safer for copy-mode with DV/PGS). MP4 is opt-in.
* Streaming is **progressive** (no real `Accept-Ranges`/`Content-Length` on mux).
  (Optional: respond `206` when client sends `Range` for better player compatibility.)

---

## 1) Project Goal

Enable Apple TV/Infuse playback of **DDP/Atmos** sidecar audio by **remuxing on the fly** with the video from Jellyfin, **without** permanently altering media files, and **without** using Jellyfin’s external-audio streaming endpoints.

---

## 2) Non-Goals

* No library management, scanning, or rewriting tags in-place.
* No transcoding (we use `-c copy` only).
* No permanent MKV/MP4 file writing (future “spooler” may be added behind a flag).
* No manipulation of video layers (e.g., stripping DV), unless a clearly named env toggle is added.

---

## 3) Runtime Assumptions

* Jellyfin is reachable at `http://10.0.0.50:8096` (configurable via `JF_URL`).
* Media is **bind-mounted** into the container (see `MEDIA_ROOTS` below).
* Sidecar naming: `<video basename>.<lang>.<ext>`, e.g.
  `District 9 (2009)…-Rel1zE.en.ac3`
* Typical sidecar extensions: `eac3`, `ec3`, `ac3`.

---

## 4) Key Files

* `app.py` — the FastAPI/Starlette app + HTTPX + `ffmpeg` pipe orchestrator.

  * `proxy_stream_direct()` — verbatim passthrough stream.
  * `proxy_stream_mux_infuse()` — conditional mux path for Infuse.
  * `find_sidecar_local_by_filename()` — filename-only sidecar discovery.
  * Real-time `ffmpeg` stderr logging for diagnostics.

---

## 5) Configuration (Env Vars)

### Frequently Tweaked

| Variable              | Default                          | Meaning                                                                                    |
| --------------------- | -------------------------------- | ------------------------------------------------------------------------------------------ |
| `JF_URL`              | `http://10.0.0.50:8096`          | Jellyfin base URL                                                                          |
| `MEDIA_ROOTS`         | `/media:/media/movies:/media/TV` | Colon-sep container dirs to search adjacent sidecars                                       |
| `AUDIO_EXTS`          | `eac3,ec3,ac3,wav,flac`                   | Allowed sidecar extensions                                                                 |
| `MUX_ON`              | `true`                           | Master on/off for live mux                                                                 |

### Occasionally Tweaked

| Variable              | Default                          | Meaning                                                                                    |
| --------------------- | -------------------------------- | ------------------------------------------------------------------------------------------ |
| `MUX_CONTAINER`       | `mkv`                            | `mkv` (recommended) or `mp4`                                                               |
| `PRESERVE_TRUEHD_WITH_SIDE` | `true`                    | Keep the original TrueHD Atmos track alongside the sidecar (even when muxing)             |
| `KEEP_ORIGINAL_TRUEHD` | `false`                         | If `true`, retain the first Jellyfin audio stream alongside the sidecar copy               |
| `AUDIO_OFFSET_MS`     | `0`                              | Apply `-itsoffset` to sidecar audio (+/− ms)                                               |
| `FORCE_PARTIAL_206`   | `true`                           | Reply `206` when client sends `Range` (even though stream is progressive)                  |
| `FORCE_APPROX_RANGE`  | `true`                           | Allow `206` with an approximate `Content-Length` when the drop-size estimate is shaky      |
| `MUX_IDLE_TIMEOUT`    | `10`                            | Auto-terminate mux stream if no bytes flow for N seconds (0 disables)                     |
| `MUX_MAX_DURATION`    | `0`                             | Hard-stop mux after N seconds regardless of activity (0 disables)                         |
| `MUX_PROGRESS_INTERVAL` | `0`                           | Seconds between `[MUX] progress bytes=…` logs (0 disables logging entirely)               |
| `MUX_USE_MKVMERGE`          | `false`                   | Use `mkvmerge` (if available) for remuxing local files instead of ffmpeg (experimental)   |
| `TRIM_ON_SESSION_STOP` | `true`                         | Stop mux early when the player closes the session                                         |
| `TRIM_AGGRESSIVE_MODE` | `false`                        | Opt-in switch for periodic + post-stream heap trims (adds GC + `malloc_trim(0)`)          |
| `TRIM_AGGRESSIVE_INTERVAL` | `0`                        | Interval between background trims when aggressive mode is enabled (`0` → falls back to 180s) |
| `TRIM_AGGRESSIVE_POST_STREAM` | `true`                 | When aggressive mode is on, trim immediately after mux/direct streams                      |
| `TRIM_LARGE_RESPONSE_BYTES` | `0`                      | When aggressive mode is on, trim after proxying responses ≥ this byte threshold            |

### Rarely Touched / Diagnostics

| Variable              | Default                          | Meaning                                                                                    |
| --------------------- | -------------------------------- | ------------------------------------------------------------------------------------------ |
| `FORCE_INSECURE_CONTENT_LENGTH` | `true`               | Keep sending 206 with the original file size when estimates are unreliable                |
| `FORCE_FAKE_CONTENT_LENGTH` | `false`                 | Force 206 using the original file size for *every* mux (testing-only)                     |
| `MUX_HEADER_CAPTURE_LIMIT` | `0`                      | Capture the first N bytes of mux output to `logs/mux_samples/` (0 disables capture)       |
| `DEBUG_FFMPEG`        | `false`                          | If `true`, `ffmpeg -loglevel info` for verbose logs                                        |
| `LOG_LEVEL`           | `INFO`                           | App log level                                                                              |
| `FFMPEG`              | `ffmpeg`                         | Path to the ffmpeg binary                                                                  |
| `FFPROBE`             | `ffprobe`                        | Path to the ffprobe binary (used for stream telemetry + TrueHD heuristics)                 |

---

## 6) API Surface (Proxy Routes)

* `GET /Videos/{item_id}/stream`

  * **Infuse** + `MUX_ON=true` → live mux if sidecar present; else passthrough.
  * Others → passthrough.
* `GET /Items/{id}` / `/PlaybackInfo` / `/MediaSegments/{id}` et al. → **proxied** to Jellyfin with auth headers preserved.
* `GET /healthz` → `{ "ok": true }`.

---

## 7) Sidecar Discovery

* We **do not** use Jellyfin’s external audio DeliveryUrl.
* We derive the **video file path** from `Items/{id}?fields=Path` and then:

  1. Construct `<basename>.<lang>.<ext>` candidates.
  2. Check **sibling directories** mapped under `MEDIA_ROOTS` (case-insensitive).
  3. Limited recursive glob as a last resort.

**Do not** change the naming convention unless you also expand the pattern list.

---

## 8) Streaming Behaviour

* **Direct path** copies Jellyfin’s headers and streams via HTTPX **inside** the generator (prevents `httpx.StreamClosed`).
* **Mux path** runs `ffmpeg` with:

  * `-map 0 -map -0:a -map 1:a:0` (keep all from main video except its audios; inject sidecar as `a:0` default).
  * `-c copy` only.
  * MKV flags to reduce interleave issues; MP4 uses fragmented options.
* We **omit** real `Content-Length` and true `Accept-Ranges` for mux.
  Optionally reply **206** when the client sends `Range` (`FORCE_PARTIAL_206=true`).
  `FORCE_APPROX_RANGE=true` forces a best-effort `Content-Range` when the size estimate is fuzzy.

---

## 9) Logging & Diagnostics

Look for these lines:

* `[MUX] remux start: …` — shows sidecar picked, container, Range header, start time.
* `[MUX] TTFB=…` — time to first muxed bytes (good if < 1–2s).
* `[ffmpeg] …` — **live stderr** from ffmpeg (codec/container errors show here).
* `[MUX] stream finished bytes=… rc=…` — exit code & byte count.
* `[MUX] head sample …` — optional capture of the first bytes of mux output into `logs/mux_samples/` with SHA-256 + hex preview, plus ready-to-run `mkvinfo` / `curl -r 0-65535 …` hints (enable via `MUX_HEADER_CAPTURE_LIMIT>0`).

Turn `DEBUG_FFMPEG=true` for richer debugging logs.

---

## 10) Quick Tests

**Direct passthrough**

```bash
curl -s -D - \
  -H "X-Emby-Token: <TOKEN>" \
  "http://10.0.0.50:9999/Videos/<ITEM_ID>/stream?Static=true" \
  -o /dev/null | sed -n '1,40p'
```

**Force mux on (per-request override)**

```bash
curl -s -D - \
  -H "X-Emby-Token: <TOKEN>" \
  -H "User-Agent: Infuse-Direct/8.2" \
  "http://10.0.0.50:9999/Videos/<ITEM_ID>/stream?Static=true&mux=1" \
  -o /dev/null | sed -n '1,80p'
```

---

## 11) Common Failure Modes & Fixes

| Symptom                                    | Likely cause                                                      | Fix                                                                                                              |
| ------------------------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Infuse “An error has occurred” immediately | Container returned 200 with no bytes; ffmpeg error                | Check `[ffmpeg]` logs; try `DEBUG_FFMPEG=true`. If MP4, consider `mkv`.                                          |
| “no external audio found near file”        | Sidecar naming doesn’t match, different lang/ext, or path mapping | Verify `<basename>.<lang>.<ext>` exactly next to the movie; adjust `AUDIO_EXTS` or `MEDIA_ROOTS` if needed.      |
| `httpx.StreamClosed` on direct path        | Context lifetime bug                                              | Direct path **must** open HTTPX stream **inside** the generator (already coded).                                 |
| AVR shows Dolby Digital (not Atmos)        | Sidecar is `.ac3` (DD 5.1), not `.ec3` (DDP/Atmos JOC)            | Use `.ec3`/`.eac3` sidecars if you need Atmos.                                                                   |
| Subtitles missing when `MUX_CONTAINER=mp4` | MP4 cannot carry PGS                                              | Use `mkv` (default).                                                                                             |
| Seeking jumps back or stalls on mux        | Progressive stream without byte-ranges                            | Current design. A future spooler (tempfile) could add proper ranges.                                             |

---

## 12) Code Style & PR Checklist (for Agents)

* **Don’t change defaults** that break existing setups. New behaviour must be opt-in via a clearly named env var.
* Keep `mux` decision **fast** and **deterministic**; never block the UI thread.
* Preserve **auth** headers verbatim on proxied calls.
* **Never** transcode by default; if adding transcoding, gate with `TRANSCODE=true` and document caveats.
* Keep logs **structured**, minimal at `INFO`, richer at `DEBUG_FFMPEG=true`.
* After completing changes, always create a git commit with a descriptive message explaining what was modified and why.
* Run quick lint before proposing patches:

  ```bash
  python -m pyflakes app.py || true
  ```
* Validate with curl tests above; include sample logs in your PR/message.

---

## 13) Roadmap (Nice-to-Have)

* **Spooler mode** (temp file) to enable true byte-range seeking (`206` with real `Content-Range`/`Content-Length`).
* Per-title overrides (e.g., force keep source audio, custom lang priority).
* Optional **DV base-layer only** toggle (for players that choke on dual-layer).
* Metrics endpoint (`/metrics`) with basic counters (TTFB, bytes, errors).

---

## 14) Security Notes

* The proxy **does not** bypass Jellyfin auth. It forwards tokens from the client.
* Avoid logging full tokens. We currently only pass them to Jellyfin or ffmpeg headers when needed.

---

## 15) Docker Compose (reference)

```yaml
version: "3.8"

services:
  infuse-atmos-adder:
    image: python:3.12-slim
    container_name: InfuseAtmosAdder
    working_dir: /app
    command:
      - bash
      - -lc
      - |
        set -e
        apt-get update && \
        apt-get install -y --no-install-recommends ffmpeg && \
        pip install fastapi uvicorn[standard] httpx && \
        exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-9999} --workers 1
    deploy:
      resources:
        limits:
          memory: ${PROXY_MEMORY_LIMIT:-10240M}
        reservations:
          memory: ${PROXY_MEMORY_RESERVATION:-512M}
    environment:
      JF_URL: http://10.0.0.50:8096
      PORT: "9999"
      LOG_LEVEL: INFO

      MUX_ON: "true"
      KEEP_ORIGINAL_TRUEHD: "false"
      AUDIO_OFFSET_MS: "0"
      MUX_CONTAINER: mkv
      FORCE_PARTIAL_206: "true"
      DEBUG_FFMPEG: "false"

      TRIM_ON_SESSION_STOP: "true"
      # Opt-in extras:
      # TRIM_AGGRESSIVE_MODE: "true"
      # TRIM_AGGRESSIVE_INTERVAL: "120"
      # TRIM_AGGRESSIVE_POST_STREAM: "true"
      # TRIM_LARGE_RESPONSE_BYTES: "67108864"

      AUDIO_EXTS: eac3,ec3,ac3
      MEDIA_ROOTS: /media/movies:/media/TV
    ports:
      - "9999:9999"
    volumes:
      - /mnt/data/media//movies:/media/movies:ro
      - /mnt/data/media//TV:/media/TV:ro
      - /mnt/apps/InfuseAtmosAdder/app.py:/app/app.py:ro
    restart: unless-stopped
```

