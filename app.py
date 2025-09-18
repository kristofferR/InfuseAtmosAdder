# app.py
# InfuseAtmosAdder live remux proxy for Infuse → Jellyfin
# - Direct-play passthrough by default
# - If enabled and client is Infuse, remux video with a filename-based sidecar
#   audio track placed next to the movie file. No Jellyfin DeliveryUrl usage.


import os
import re
import glob
import time
import json
import gc
import asyncio
import contextlib
import logging
import binascii
import hashlib
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, Iterable, Optional

import anyio
import httpx
from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse

# ----------------------------- config ---------------------------------
JF_BASE = os.getenv("JF_URL", "http://10.0.0.50:8096").rstrip("/")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
FFMPEG_BIN = os.getenv("FFMPEG", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE", "ffprobe")

_ffprobe_missing_logged = False

MUX_ON = os.getenv("MUX_ON", "true").lower() in ("1", "true", "yes", "on")
KEEP_ORIGINAL_TRUEHD = os.getenv("KEEP_ORIGINAL_TRUEHD", "false").lower() in ("1", "true", "yes", "on")
AUDIO_OFFSET_MS = int(os.getenv("AUDIO_OFFSET_MS", "0"))
MUX_CONTAINER = os.getenv("MUX_CONTAINER", "mkv").lower()  # mkv|mp4
FORCE_PARTIAL_206 = os.getenv("FORCE_PARTIAL_206", "true").lower() in ("1", "true", "yes", "on")
DEBUG_FFMPEG = os.getenv("DEBUG_FFMPEG", "false").lower() in ("1", "true", "yes", "on")
TRIM_ON_SESSION_STOP = os.getenv("TRIM_ON_SESSION_STOP", "true").lower() in ("1", "true", "yes", "on")
FORCE_APPROX_RANGE = os.getenv("FORCE_APPROX_RANGE", "true").lower() in ("1", "true", "yes", "on")
FORCE_INSECURE_CONTENT_LENGTH = os.getenv("FORCE_INSECURE_CONTENT_LENGTH", "true").lower() in ("1", "true", "yes", "on")
FORCE_FAKE_CONTENT_LENGTH = os.getenv("FORCE_FAKE_CONTENT_LENGTH", "false").lower() in ("1", "true", "yes", "on")
USE_MKVMERGE = os.getenv("MUX_USE_MKVMERGE", "false").lower() in ("1", "true", "yes", "on")
MKVMERGE_BIN = os.getenv("MKVMERGE", "mkvmerge")
PRESERVE_TRUEHD_WITH_SIDE = os.getenv("PRESERVE_TRUEHD_WITH_SIDE", "true").lower() in ("1", "true", "yes", "on")

TRIM_AGGRESSIVE_MODE = os.getenv("TRIM_AGGRESSIVE_MODE", "false").lower() in ("1", "true", "yes", "on")
try:
    _trim_interval_raw = float(os.getenv("TRIM_AGGRESSIVE_INTERVAL", "0"))
    if _trim_interval_raw < 0:
        _trim_interval_raw = 0.0
except ValueError:
    _trim_interval_raw = 0.0
TRIM_AGGRESSIVE_INTERVAL = 0.0
if TRIM_AGGRESSIVE_MODE:
    TRIM_AGGRESSIVE_INTERVAL = _trim_interval_raw or 180.0

TRIM_AGGRESSIVE_POST_STREAM = False
if TRIM_AGGRESSIVE_MODE:
    TRIM_AGGRESSIVE_POST_STREAM = os.getenv("TRIM_AGGRESSIVE_POST_STREAM", "true").lower() in ("1", "true", "yes", "on")

try:
    TRIM_LARGE_RESPONSE_BYTES = int(os.getenv("TRIM_LARGE_RESPONSE_BYTES", "0"))
    if TRIM_LARGE_RESPONSE_BYTES < 0:
        TRIM_LARGE_RESPONSE_BYTES = 0
except ValueError:
    TRIM_LARGE_RESPONSE_BYTES = 0
if not TRIM_AGGRESSIVE_MODE:
    TRIM_LARGE_RESPONSE_BYTES = 0

TRIM_ENABLED = TRIM_ON_SESSION_STOP or TRIM_AGGRESSIVE_MODE
try:
    MUX_PROGRESS_INTERVAL = float(os.getenv("MUX_PROGRESS_INTERVAL", "0"))
    if MUX_PROGRESS_INTERVAL < 0:
        MUX_PROGRESS_INTERVAL = 0.0
except ValueError:
    MUX_PROGRESS_INTERVAL = 0.0
try:
    MUX_IDLE_TIMEOUT = float(os.getenv("MUX_IDLE_TIMEOUT", "10"))
    if MUX_IDLE_TIMEOUT < 0:
        MUX_IDLE_TIMEOUT = 0.0
except ValueError:
    MUX_IDLE_TIMEOUT = 10.0
try:
    MUX_MAX_DURATION = float(os.getenv("MUX_MAX_DURATION", "0"))
    if MUX_MAX_DURATION < 0:
        MUX_MAX_DURATION = 0.0
except ValueError:
    MUX_MAX_DURATION = 0.0
try:
    MUX_PREROLL_BYTES = int(os.getenv("MUX_PREROLL_BYTES", "0"))
    if MUX_PREROLL_BYTES < 0:
        MUX_PREROLL_BYTES = 0
except ValueError:
    MUX_PREROLL_BYTES = 0

# Language codes to accept in sidecar filenames: "<base>.<lang>.<ext>"
AUDIO_EXTS = [t.strip().lower().lstrip(".") for t in os.getenv("AUDIO_EXTS", "eac3,ec3,ac3").split(",") if t.strip()]

# Where media is mounted **inside the container** (colon-separated)
MEDIA_ROOTS = [p for p in os.getenv("MEDIA_ROOTS", "/media:/media/movies:/media/TV").split(":") if p]

# ----------------------------- logging --------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_SMART_QUOTE_TABLE = {
    ord("\u2018"): None,  # left single quote
    ord("\u2019"): None,  # right single quote
    ord("\u201c"): None,  # left double quote
    ord("\u201d"): None,  # right double quote
}

_DEFAULT_LOG_DIR = os.path.join(_BASE_DIR, "logs")
_raw_log_dir = os.getenv("LOG_DIR", "")
_raw_log_dir_stripped = _raw_log_dir.strip()
_sanitized_log_dir = (
    _raw_log_dir_stripped.translate(_SMART_QUOTE_TABLE).strip()
    if _raw_log_dir_stripped
    else ""
)
if _sanitized_log_dir:
    LOG_DIR = _sanitized_log_dir
    _log_dir_note = None
else:
    LOG_DIR = _DEFAULT_LOG_DIR
    _log_dir_note = None if not _raw_log_dir else (
        f"[LOGGING] LOG_DIR invalid after sanitization; falling back to default raw={_raw_log_dir!r}"
    )

LOG_FILE = os.path.join(LOG_DIR, "proxy.log")
MUX_SAMPLE_DIR = os.path.join(LOG_DIR, "mux_samples")
try:
    MUX_HEADER_CAPTURE_LIMIT = int(os.getenv("MUX_HEADER_CAPTURE_LIMIT", "0"))
    if MUX_HEADER_CAPTURE_LIMIT < 0:
        MUX_HEADER_CAPTURE_LIMIT = 0
except ValueError:
    MUX_HEADER_CAPTURE_LIMIT = 0
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

_handlers: list[logging.Handler] = []

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
_handlers.append(stream_handler)

file_handler: Optional[logging.Handler] = None
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight", backupCount=7, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    _handlers.append(file_handler)
except OSError:
    file_handler = None

logging.basicConfig(level=LOG_LEVEL, handlers=_handlers, force=True)
log = logging.getLogger("proxy")

if file_handler is not None:
    log.info("[LOGGING] file logging enabled path=%s", LOG_FILE)
else:
    log.warning("[LOGGING] file logging disabled; falling back to stdout only log_dir=%s", LOG_DIR)

if _log_dir_note:
    log.warning(_log_dir_note)


def _ensure_logger_handlers(name: str, propagate: Optional[bool] = None) -> None:
    logger = logging.getLogger(name)
    for handler in _handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)
    if propagate is not None:
        logger.propagate = propagate
    if logger.level == logging.NOTSET:
        logger.setLevel(LOG_LEVEL)


for logger_name in (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "uvicorn.asgi",
    "uvicorn.lifespan",
):
    _ensure_logger_handlers(logger_name, propagate=False)

_ensure_logger_handlers("httpx")


# ----------------------------- app ------------------------------------
app = FastAPI()


@app.on_event("startup")
async def _start_trim_scheduler() -> None:
    if not (TRIM_ENABLED and TRIM_AGGRESSIVE_MODE and TRIM_AGGRESSIVE_INTERVAL > 0):
        return
    global _periodic_trim_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover
        return
    if _periodic_trim_task is None or _periodic_trim_task.done():
        _periodic_trim_task = loop.create_task(_periodic_trim_loop(TRIM_AGGRESSIVE_INTERVAL))
        log.info("[TRIM] periodic trim enabled interval=%.1fs", TRIM_AGGRESSIVE_INTERVAL)


@app.on_event("shutdown")
async def _stop_trim_scheduler() -> None:
    global _periodic_trim_task
    if _periodic_trim_task is None:
        return
    _periodic_trim_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _periodic_trim_task
    _periodic_trim_task = None
    log.info("[TRIM] periodic trim stopped")

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade"
}

COPY_HEADERS = {
    "content-type", "content-length", "content-range",
    "accept-ranges", "etag", "last-modified", "cache-control", "server",
    "x-emby-authorization", "x-powered-by", "date", "expires", "pragma",
    "content-disposition"
}

# ----------------------------- helpers --------------------------------
def _filter_headers(src: Iterable[tuple[str, str]], allowed_lower: set[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in src:
        kl = k.lower()
        if kl in HOP_BY_HOP:
            continue
        if kl in allowed_lower:
            out[k] = v
    return out

def _build_jf_url(path: str, query: str = "") -> str:
    if query:
        return f"{JF_BASE}{path}?{query}"
    return f"{JF_BASE}{path}"

def _pass_through_auth(req: Request) -> Dict[str, str]:
    h = {}
    for name in (
        "Authorization",
        "X-Emby-Authorization",
        "X-Emby-Token",
        "X-MediaBrowser-Token",
        "User-Agent",
        "Range",
        "Accept",
        "Accept-Encoding",
        "Content-Type",
    ):
        if name in req.headers:
            h[name] = req.headers[name]
    return h

def _extract_token(req: Request) -> Optional[str]:
    for k in ("X-Emby-Token", "X-MediaBrowser-Token"):
        if k in req.headers:
            return req.headers[k]
    auth = req.headers.get("Authorization") or req.headers.get("X-Emby-Authorization")
    if auth:
        m = re.search(r'Token="([^"]+)"', auth)
        if m:
            return m.group(1)
    return None

def _build_ffmpeg_headers(req: Request, token: Optional[str]) -> Optional[str]:
    lines: list[str] = []
    if token:
        lines.append(f"X-Emby-Token: {token}")
    for header_name in ("Authorization", "X-Emby-Authorization"):
        value = req.headers.get(header_name)
        if value:
            lines.append(f"{header_name}: {value}")
    if not lines:
        return None
    return "\r\n".join(lines) + "\r\n"


def _hex_prefix(data: bytes, length: int = 64) -> str:
    if not data:
        return ""
    return binascii.hexlify(data[:length]).decode(errors="ignore")


def _read_rss_bytes() -> Optional[int]:
    if os.name != "posix":
        return None
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.readline().split()
        if len(parts) < 2:
            return None
        rss_pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return rss_pages * page_size
    except (OSError, ValueError):
        return None


def _format_bytes(num: Optional[int]) -> str:
    if num is None:
        return "?"
    if num == 0:
        return "0B"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{int(num)}B"


def _format_delta(num: Optional[int]) -> str:
    if num is None:
        return "?"
    if num == 0:
        return "0"
    if num < 0:
        return f"-{_format_bytes(-num)}"
    return f"+{_format_bytes(num)}"


def _maybe_malloc_trim() -> Optional[int]:
    if os.name != "posix":
        return None
    try:
        import ctypes
    except ImportError:
        return None

    libc_candidates = (None, "libc.so.6", "libc.so", "libSystem.dylib", "libSystem.B.dylib")
    for candidate in libc_candidates:
        try:
            libc = ctypes.CDLL(candidate) if candidate is not None else ctypes.CDLL(None)
        except OSError:
            continue
        trim_fn = getattr(libc, "malloc_trim", None)
        if trim_fn is None:
            continue
        try:
            return int(trim_fn(0))
        except Exception:  # pragma: no cover
            return None
    return None


def _perform_aggressive_trim(reason: str) -> None:
    start = time.monotonic()
    rss_before = _read_rss_bytes()
    collected = gc.collect()
    trim_result = _maybe_malloc_trim()
    rss_after = _read_rss_bytes()
    delta = None
    if rss_before is not None and rss_after is not None:
        delta = rss_after - rss_before
    log_fn = log.info if reason != "periodic" else log.debug
    log_fn(
        "[TRIM] reason=%s gc_collected=%d malloc_trim=%s rss_before=%s rss_after=%s delta=%s elapsed=%.3fs",
        reason,
        collected,
        trim_result if trim_result is not None else "n/a",
        _format_bytes(rss_before),
        _format_bytes(rss_after),
        _format_delta(delta),
        time.monotonic() - start,
    )


_trim_lock: Optional[asyncio.Lock] = None
_periodic_trim_task: Optional[asyncio.Task] = None


async def _trim_memory(reason: str) -> None:
    if not TRIM_ENABLED:
        return
    global _trim_lock
    if _trim_lock is None:
        _trim_lock = asyncio.Lock()
    async with _trim_lock:
        try:
            await anyio.to_thread.run_sync(_perform_aggressive_trim, reason)
        except Exception:  # pragma: no cover
            log.exception("[TRIM] aggressive trim failed reason=%s", reason)


def _trigger_aggressive_trim(reason: str) -> None:
    if not TRIM_ENABLED:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _perform_aggressive_trim(reason)
        return

    async def _runner() -> None:
        try:
            await _trim_memory(reason)
        except Exception:  # pragma: no cover
            log.exception("[TRIM] background trim failed reason=%s", reason)

    loop.create_task(_runner())


async def _periodic_trim_loop(interval: float) -> None:
    try:
        while True:
            await asyncio.sleep(interval)
            await _trim_memory("periodic")
    except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
        raise
    except Exception:  # pragma: no cover
        log.exception("[TRIM] periodic trim loop failed")


async def _body_bytes(req: Request) -> bytes:
    try:
        return await req.body()
    except Exception:
        return b""

async def jf_json(path: str, req: Request, query: str = "") -> dict:
    url = _build_jf_url(path, query)
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(url, headers=_pass_through_auth(req))
        r.raise_for_status()
        return r.json()

# ------------------------ generic small proxy --------------------------
async def proxy_small(req: Request, jf_path: str) -> Response:
    q = req.url.query
    jf_url = _build_jf_url(jf_path, q)
    data = await _body_bytes(req)

    t0 = time.monotonic()
    body_len = len(data)

    log.info(
        "[PASS] %s %s -> JF %s body=%dB client=%s",
        req.method,
        req.url.path,
        jf_url,
        body_len,
        req.client.host if req.client else "?",
    )

    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.request(
            req.method, jf_url,
            headers=_pass_through_auth(req),
            content=data if data else None,
        )

    elapsed = time.monotonic() - t0
    content_length = r.headers.get("Content-Length")
    body_bytes = len(r.content) if r.content is not None else 0

    log.info(
        "[PASS<-JF] %s %s status=%s body=%dB header_CL=%s elapsed=%.3fs",
        req.method,
        req.url.path,
        r.status_code,
        body_bytes,
        content_length,
        elapsed,
    )

    if TRIM_LARGE_RESPONSE_BYTES and body_bytes >= TRIM_LARGE_RESPONSE_BYTES:
        _trigger_aggressive_trim("large_response")

    if (
        TRIM_ON_SESSION_STOP
        and req.method.upper() == "POST"
        and jf_path == "/Sessions/Playing/Stopped"
        and r.status_code == 204
    ):
        _trigger_aggressive_trim("session_stop")

    headers = _filter_headers(r.headers.items(), COPY_HEADERS)
    return Response(content=r.content, status_code=r.status_code, headers=headers)

# --------------------------- stream proxy ------------------------------
async def proxy_stream_direct(req: Request, jf_path: str) -> StreamingResponse:
    """
    Stream verbatim from Jellyfin, keeping upstream stream open for lifetime
    of response (prevents httpx.StreamClosed).
    """
    jf_url = _build_jf_url(jf_path, req.url.query)
    log.info("[STREAM->JF] %s '%s' UA='%s' Range='%s' URL='%s'",
             req.method, req.url.path, req.headers.get("User-Agent"),
             req.headers.get("Range"), jf_url)

    # Preflight to copy status + headers
    status = 200
    out_headers: Dict[str, str] = {}
    async with httpx.AsyncClient(timeout=None) as c:
        try:
            head = await c.head(jf_url, headers=_pass_through_auth(req))
            if head.status_code in (200, 206):
                status = head.status_code
                out_headers = _filter_headers(head.headers.items(), COPY_HEADERS)
            else:
                async with c.stream("GET", jf_url, headers=_pass_through_auth(req)) as r0:
                    status = r0.status_code
                    out_headers = _filter_headers(r0.headers.items(), COPY_HEADERS)
        except httpx.RequestError:
            async with c.stream("GET", jf_url, headers=_pass_through_auth(req)) as r0:
                status = r0.status_code
                out_headers = _filter_headers(r0.headers.items(), COPY_HEADERS)

    async def body_iter():
        total = 0
        first = True
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=None) as c2:
                async with c2.stream("GET", jf_url, headers=_pass_through_auth(req)) as r:
                    log.info("[STREAM<-JF] %s CT='%s' CL='%s' CR='%s'",
                             r.status_code, r.headers.get("Content-Type"),
                             r.headers.get("Content-Length"), r.headers.get("Content-Range"))
                    with contextlib.suppress(
                        httpx.ReadError, anyio.EndOfStream, anyio.ClosedResourceError, asyncio.CancelledError
                    ):
                        async for chunk in r.aiter_raw():
                            if not chunk:
                                continue
                            if first:
                                log.info("[STREAM] TTFB=%.3fs first_chunk=%d bytes", time.monotonic() - t0, len(chunk))
                                first = False
                            total += len(chunk)
                            yield chunk
        finally:
            log.info("[STREAM] finished forwarding total=%d bytes", total)
            if TRIM_AGGRESSIVE_POST_STREAM:
                with contextlib.suppress(asyncio.CancelledError):
                    await _trim_memory("post_direct_stream")

    return StreamingResponse(body_iter(), status_code=status, headers=out_headers)

# ---------------------- filename-based sidecars ------------------------
def _map_host_to_container_dir(item_path: str) -> list[str]:
    """
    Map the Jellyfin host path to container directories by probing MEDIA_ROOTS
    for matching suffixes. We progressively drop leading segments from the
    parent dir until we find real directories, which handles mounts like
    "/movies" → "/media/movies" without duplicating path pieces.
    """
    if not item_path:
        return []

    host_dir = os.path.dirname(os.path.normpath(item_path))
    host_parts = [p for p in host_dir.split(os.sep) if p]
    if not host_parts:
        return MEDIA_ROOTS

    candidates: list[str] = []
    max_tail = min(len(host_parts), 6)
    tail_options: list[list[str]] = []
    for take in range(max_tail, 0, -1):
        tail = host_parts[-take:]
        if tail and tail not in tail_options:
            tail_options.append(tail)

    for root in MEDIA_ROOTS:
        norm_root = os.path.normpath(root)
        if os.path.isdir(norm_root) and norm_root not in candidates:
            candidates.append(norm_root)
        for tail in tail_options:
            for skip in range(len(tail)):
                sub_tail = tail[skip:]
                cand = os.path.join(norm_root, *sub_tail) if sub_tail else norm_root
                if os.path.isdir(cand):
                    if cand not in candidates:
                        candidates.append(cand)
                    break

    return candidates or MEDIA_ROOTS


def _resolve_container_file(item_path: str) -> Optional[str]:
    """Attempt to map a Jellyfin host path to an accessible container path."""
    if not item_path:
        return None
    filename = os.path.basename(item_path)
    for directory in _map_host_to_container_dir(item_path):
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return None

def find_sidecar_local_by_filename(item_path: str) -> Optional[str]:
    """
    Pure filename-based: look for "<basename>[.<anything>].<ext>" next to the movie.
    Returns absolute container path or None.
    """
    if not item_path:
        return None

    base = os.path.splitext(os.path.basename(item_path))[0]
    base_lower = base.lower()
    candidates_dirs = _map_host_to_container_dir(item_path)

    def _is_matching(filename: str) -> bool:
        stem, ext = os.path.splitext(filename)
        if not ext:
            return False
        ext_lower = ext.lstrip(".").lower()
        if ext_lower not in AUDIO_EXTS:
            return False
        stem_lower = stem.lower()
        return stem_lower == base_lower or stem_lower.startswith(f"{base_lower}.")

    # 1) Direct same-dir checks (sorted for determinism)
    for directory in candidates_dirs:
        try:
            names = sorted(os.listdir(directory))
        except Exception:
            names = []
        for name in names:
            full_path = os.path.join(directory, name)
            if not os.path.isfile(full_path):
                continue
            if _is_matching(name):
                return full_path

    # 2) Limited recursive glob under configured roots
    escaped_base = glob.escape(base)
    for root in MEDIA_ROOTS:
        pattern = os.path.join(root, "**", f"{escaped_base}*")
        try:
            for hit in glob.iglob(pattern, recursive=True):
                if not os.path.isfile(hit):
                    continue
                if _is_matching(os.path.basename(hit)):
                    return hit
        except re.error:
            continue
    return None


_RANGE_RE = re.compile(r"bytes=([0-9]+)-([0-9]*)", re.IGNORECASE)

def _is_initial_byte_range(range_header: str) -> bool:
    if not range_header:
        return False
    m = _RANGE_RE.match(range_header.strip())
    return bool(m and m.group(1) == "0")

COMMENTARY_KEYWORDS = (
    "commentary",
    "commentator",
    "director",
    "writer",
    "producer",
    "roundtable",
    "discussion",
    "isolated score",
    "isolated music",
    "isolated track",
    "podcast",
    "interview",
)

def _normalize_language(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = value.strip().lower()
    return lowered or None

def _first_text(*values: Optional[str]) -> Optional[str]:
    for val in values:
        if isinstance(val, str):
            stripped = val.strip()
            if stripped:
                return stripped
    return None

def _coerce_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _looks_like_commentary(title: Optional[str]) -> bool:
    if not title:
        return False
    lowered = title.lower()
    return any(keyword in lowered for keyword in COMMENTARY_KEYWORDS)

def _enumerate_audio_streams(meta: dict) -> list[dict]:
    primary = (meta.get("MediaSources") or [{}])[0]
    streams = primary.get("MediaStreams") or meta.get("MediaStreams") or []
    audio: list[dict] = []
    order = 0
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        if (stream.get("Type") or "").lower() != "audio":
            continue
        raw_index = stream.get("Index")
        ff_index = raw_index if isinstance(raw_index, int) and raw_index >= 0 else None
        # Always address audio streams by their order for deterministic ffmpeg mapping.
        map_spec = f"0:a:{order}"
        if ff_index is not None:
            map_display = f"input0:{ff_index} (a:{order})"
        else:
            map_display = f"input0:a:{order}"
        audio.append({
            "stream": stream,
            "order": order,
            "map_spec": map_spec,
            "map_display": map_display,
            "ff_index": ff_index,
        })
        order += 1
    return audio

def _combine_audio_details(audio_streams: list[dict], probe_streams: list[dict]) -> list[dict]:
    details: list[dict] = []
    if probe_streams and len(probe_streams) != len(audio_streams):
        log.debug(
            "[FFPROBE] audio stream count mismatch jellyfin=%d ffprobe=%d",
            len(audio_streams), len(probe_streams),
        )

    for idx, base in enumerate(audio_streams):
        probe = probe_streams[idx] if idx < len(probe_streams) else {}
        stream = base.get("stream") or {}

        probe_tags_raw = probe.get("tags") or {}
        probe_tags = {str(k).lower(): v for k, v in probe_tags_raw.items()}
        probe_bit_rate = _coerce_int(probe.get("bit_rate"))
        disposition = probe.get("disposition") or {}

        codec_name = (probe.get("codec_name") or stream.get("Codec") or "").lower()
        language = _normalize_language(
            _first_text(stream.get("Language"), probe_tags.get("language"), probe_tags.get("lang"))
        )
        title = _first_text(stream.get("DisplayTitle"), stream.get("Title"), probe_tags.get("title")) or ""
        channels = _coerce_int(probe.get("channels")) or _coerce_int(stream.get("Channels"))
        channel_layout = _first_text(stream.get("ChannelLayout"), probe.get("channel_layout"))
        is_default = bool(stream.get("IsDefault") or stream.get("IsForced") or disposition.get("default"))
        commentary_flag = bool(stream.get("IsCommentary") or stream.get("Commentary"))
        is_commentary = commentary_flag or _looks_like_commentary(title)

        detail = {
            "base": base,
            "stream": stream,
            "probe": probe,
            "tags": probe_tags,
            "probe_bit_rate": probe_bit_rate,
            "codec_name": codec_name,
            "language": language,
            "title": title,
            "title_lower": title.lower(),
            "channels": channels,
            "channel_layout": channel_layout,
            "is_default": is_default,
            "is_commentary": is_commentary,
            "is_truehd": "truehd" in codec_name,
        }
        details.append(detail)
    return details

def _select_primary_audio_detail(details: list[dict]) -> Optional[dict]:
    if not details:
        return None

    truehd_details = [d for d in details if d["is_truehd"]]
    candidates = truehd_details or details

    non_commentary = [d for d in candidates if not d["is_commentary"]]
    if non_commentary:
        candidates = non_commentary

    def _score(info: dict) -> tuple:
        title_lower = info.get("title_lower") or ""
        language = info.get("language") or ""
        channels = info.get("channels") or 0
        order = info.get("base", {}).get("order", 0)
        return (
            1 if info.get("is_default") else 0,
            1 if "atmos" in title_lower or "dolby" in title_lower else 0,
            1 if language in {"eng", "en"} else 0,
            channels,
            -order,
        )

    selected = max(candidates, key=_score)
    return selected

def _estimate_stream_bytes(
    stream: dict,
    fallback_ticks: Optional[int],
    detail: Optional[dict] = None,
) -> tuple[Optional[int], bool, str]:
    if detail:
        tags = detail.get("tags") or {}
        for key, value in tags.items():
            if key.startswith("number_of_bytes"):
                tag_bytes = _coerce_int(value)
                if tag_bytes and tag_bytes > 0:
                    return tag_bytes, True, f"tag:{key}"
        probe_size = _coerce_int(detail.get("probe", {}).get("size"))
        if probe_size and probe_size > 0:
            return probe_size, True, "probe:size"

    size_val = _coerce_int(stream.get("Size"))
    if size_val and size_val > 0:
        return size_val, True, "media_stream.size"

    bitrate = _coerce_int(stream.get("BitRate"))
    if detail and not bitrate:
        bitrate = detail.get("probe_bit_rate") or None
        if not bitrate:
            tags = detail.get("tags") or {}
            for key, value in tags.items():
                if key.startswith("bps"):
                    bitrate = _coerce_int(value)
                    if bitrate:
                        break
    if not bitrate or bitrate <= 0:
        return None, False, "no-bitrate"

    ticks = _coerce_int(stream.get("RunTimeTicks")) or fallback_ticks
    if not ticks:
        return None, False, "no-runtime"
    try:
        ticks_val = int(ticks)
    except (TypeError, ValueError):
        return None, False, "bad-runtime"
    seconds = ticks_val / 10_000_000
    return int((bitrate / 8) * seconds), False, "bitrate"


def _estimate_muxed_size(
    meta: dict,
    item_path: str,
    sidecar_path: str,
    removed_streams: Optional[list[tuple[dict, Optional[dict]]]] = None,
) -> tuple[Optional[int], bool, Optional[int]]:
    """Best-effort estimate of muxed stream size for Content-Range/Length."""
    primary = (meta.get("MediaSources") or [{}])[0]
    base_size = primary.get("Size")

    if isinstance(base_size, str):
        try:
            base_size = int(base_size)
        except ValueError:
            base_size = None

    if base_size is None and item_path:
        sidecar_dir = os.path.dirname(sidecar_path)
        base_candidate = os.path.join(sidecar_dir, os.path.basename(item_path))
        try:
            base_size = os.path.getsize(base_candidate)
        except OSError:
            base_size = None

    try:
        sidecar_size = os.path.getsize(sidecar_path)
    except OSError:
        return None, False, base_size

    if base_size is None:
        return None, False, base_size

    estimated = base_size
    removed_streams = removed_streams or []
    size_reliable = True

    if KEEP_ORIGINAL_TRUEHD:
        estimated += sidecar_size
    elif removed_streams:
        timeline_ticks = primary.get("RunTimeTicks") or meta.get("RunTimeTicks")
        drop_bytes = 0
        for stream, detail in removed_streams:
            stream_bytes, bytes_are_reliable, source = _estimate_stream_bytes(stream, timeline_ticks, detail)
            if stream_bytes is None:
                log.info(
                    "[MUX] unable to estimate drop bytes for order=%s reason=%s",
                    stream.get("Index"),
                    source,
                )
                return None, False
            if not bytes_are_reliable:
                size_reliable = False
                log.info(
                    "[MUX] drop-bytes estimate via %s order=%s bytes≈%s",
                    source,
                    stream.get("Index"),
                    _format_bytes(stream_bytes),
                )
            drop_bytes += stream_bytes
        estimated = estimated - drop_bytes + sidecar_size
    else:
        estimated += sidecar_size

    if estimated <= 0:
        return None, False, base_size
    return int(estimated), size_reliable, base_size

# ---------------------- ffmpeg helpers ---------------------------------
def _ff_loglvl() -> str:
    return "info" if DEBUG_FFMPEG else "warning"

async def _pump_stderr(name: str, stream: asyncio.StreamReader):
    """Log ffmpeg stderr lines in real time."""
    if not stream:
        return
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            s = line.decode(errors="ignore").rstrip()
            if s:
                log.warning("[%s] %s", name, s)
    except Exception:
        pass

async def _ffprobe_audio_streams(main_url: str, header_blob: Optional[str]) -> list[dict]:
    global _ffprobe_missing_logged
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-select_streams", "a",
        "-show_entries",
        "stream=index,codec_name,codec_type,channels,channel_layout,bit_rate:stream_tags:stream_disposition=default",
        "-of", "json",
    ]
    if header_blob:
        cmd += ["-headers", header_blob]
    cmd += ["-i", main_url]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        if not _ffprobe_missing_logged:
            log.warning(
                "[FFPROBE] binary '%s' not found; unable to identify TrueHD streams."
                " Install ffmpeg/ffprobe or set FFPROBE path to avoid commentary swaps.",
                FFPROBE_BIN,
            )
            _ffprobe_missing_logged = True
        return []
    except Exception as exc:  # pragma: no cover
        log.warning("[FFPROBE] failed to spawn '%s': %r", FFPROBE_BIN, exc)
        return []

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.communicate(), timeout=1.0)
        log.debug("[FFPROBE] timed out probing %s", main_url)
        return []

    if proc.returncode != 0:
        err_str = stderr.decode(errors="ignore").strip()
        log.debug("[FFPROBE] non-zero exit code=%s err='%s'", proc.returncode, err_str)
        return []

    try:
        payload = json.loads(stdout.decode() or "{}")
    except json.JSONDecodeError:
        log.debug("[FFPROBE] invalid JSON response")
        return []

    streams = payload.get("streams")
    if not isinstance(streams, list):
        return []
    return streams

# ---------------------- sidecar audio mux (optional) -------------------
async def proxy_stream_mux_infuse(req: Request, item_id: str) -> Optional[StreamingResponse]:
    """
    Only filename-based sidecar lookup. We fetch /Items/{id}?fields=Path,Name
    to learn the video file path; we do NOT use any DeliveryUrl/API for audio.
    """
    # Per-request override: ?mux=0 or ?mux=1 (default: env/global)
    qs = dict(req.query_params)
    if "mux" in qs:
        enabled = qs.get("mux", "1").lower() not in ("0", "false", "no")
    else:
        enabled = MUX_ON
    if not enabled:
        return None

    ua = (req.headers.get("User-Agent") or "").lower()
    if "infuse" not in ua:
        return None  # only do this for Infuse clients

    # Get the source video path from Jellyfin (no external audio API usage)
    try:
        meta = await jf_json(f"/Items/{item_id}", req, "fields=Path,Name,MediaSources")
    except Exception as e:
        log.warning("[MUX] failed to get item path: %r", e)
        return None

    item_path = (meta.get("Path")
                 or ((meta.get("MediaSources") or [{}])[0].get("Path"))
                 or "")
    if not item_path:
        log.info("[MUX] cannot determine item path; fallback to direct")
        return None

    sidecar = find_sidecar_local_by_filename(item_path)
    if not sidecar:
        log.info("[MUX] no external audio found near file; fallback to direct")
        return None

    audio_streams = _enumerate_audio_streams(meta)

    # Build the original static video URL mirroring client request (keep query intact)
    main_url = _build_jf_url(f"/Videos/{item_id}/stream", req.url.query)

    token = _extract_token(req)
    http_headers = _build_ffmpeg_headers(req, token)

    probe_streams: list[dict] = []
    local_video_path = _resolve_container_file(item_path)
    use_local_file = bool(local_video_path)
    if audio_streams:
        probe_target = local_video_path if use_local_file else main_url
        probe_headers = None if use_local_file else http_headers
        probe_streams = await _ffprobe_audio_streams(probe_target, probe_headers)

    audio_details = _combine_audio_details(audio_streams, probe_streams)
    detail_map = {detail["base"]["order"]: detail for detail in audio_details if detail.get("base")}

    if audio_details:
        for detail in audio_details:
            base = detail["base"]
            log.debug(
                "[MUX] source audio order=%d codec=%s channels=%s layout=%s lang=%s default=%s commentary=%s title='%s'",
                base["order"],
                detail.get("codec_name") or (base["stream"].get("Codec") if base else ""),
                detail.get("channels") or "?",
                detail.get("channel_layout") or "",
                detail.get("language") or "und",
                detail.get("is_default"),
                detail.get("is_commentary"),
                detail.get("title"),
            )

    primary_detail = _select_primary_audio_detail(audio_details)
    primary_audio: Optional[dict] = primary_detail.get("base") if primary_detail else None
    if primary_audio is None and audio_streams:
        primary_audio = audio_streams[0]

    sidecar_language: Optional[str] = None
    if primary_detail and primary_detail.get("language"):
        sidecar_language = primary_detail["language"]
    elif primary_audio is not None:
        raw_lang = primary_audio["stream"].get("Language")
        if isinstance(raw_lang, str):
            lang = raw_lang.strip()
            if lang:
                sidecar_language = lang.lower()

    removed_streams: list[tuple[dict, Optional[dict]]] = []
    drop_primary = not KEEP_ORIGINAL_TRUEHD
    if drop_primary and PRESERVE_TRUEHD_WITH_SIDE and primary_detail and primary_detail.get("is_truehd"):
        drop_primary = False
        log.info("[MUX] preserving source TrueHD alongside sidecar due to PRESERVE_TRUEHD_WITH_SIDE")

    if drop_primary and primary_audio is not None:
        target = primary_audio["stream"]
        detail = detail_map.get(primary_audio["order"])
        removed_streams.append((target, detail))
        codec_name = detail.get("codec_name") if detail else target.get("Codec")
        language = (detail.get("language") if detail else target.get("Language")) or "und"
        channels = detail.get("channels") if detail else target.get("Channels")
        title = (detail.get("title") if detail else target.get("DisplayTitle") or target.get("Title") or "")
        log.info(
            "[MUX] replacing source audio order=%d codec=%s lang=%s channels=%s title='%s'",
            primary_audio["order"],
            codec_name,
            language,
            channels if channels is not None else "?",
            title,
        )
    elif primary_audio is None:
        log.info("[MUX] source video has no audio streams; sidecar becomes only track")

    estimated_size, size_is_reliable, base_file_size = _estimate_muxed_size(meta, item_path, sidecar, removed_streams)

    effective_size = estimated_size
    fallback_cl_used = False
    if base_file_size and base_file_size > 0:
        if FORCE_FAKE_CONTENT_LENGTH:
            effective_size = base_file_size
            fallback_cl_used = True
            log.info(
                "[MUX] forcing Content-Length to original size=%s (estimated=%s reliable=%s)",
                _format_bytes(base_file_size),
                _format_bytes(estimated_size),
                size_is_reliable,
            )
        elif FORCE_INSECURE_CONTENT_LENGTH:
            if effective_size is None or base_file_size >= effective_size or not size_is_reliable:
                effective_size = base_file_size
                fallback_cl_used = True
                log.info(
                    "[MUX] using fallback Content-Length size=%s (estimated=%s reliable=%s)",
                    _format_bytes(base_file_size),
                    _format_bytes(estimated_size),
                    size_is_reliable,
                )

    # Honor StartTimeTicks/StartTimeMs for initial seek
    start_ticks = 0
    for k in ("StartTimeTicks", "startTimeTicks"):
        if k in qs:
            try:
                start_ticks = int(qs[k]); break
            except Exception:
                pass
    if (start_ticks == 0) and ("StartTimeMs" in qs):
        try:
            start_ticks = int(qs["StartTimeMs"]) * 10_000
        except Exception:
            pass
    start_seconds = start_ticks / 10_000_000 if start_ticks else 0.0

    use_mkvmerge_pipeline = USE_MKVMERGE and use_local_file and MUX_CONTAINER == "mkv"

    if use_mkvmerge_pipeline:
        if KEEP_ORIGINAL_TRUEHD:
            log.warning("[MUX] MKVMERGE pipeline does not support KEEP_ORIGINAL_TRUEHD; ignoring")
        mkvmerge_cmd = [MKVMERGE_BIN, "-o", "-", "--no-audio", local_video_path]
        track_lang = sidecar_language or "und"
        mkvmerge_cmd += ["--language", f"0:{track_lang}", "--default-track", "0:yes", sidecar]
        log.info("[MUX] using mkvmerge pipeline cmd=%s", mkvmerge_cmd)
        content_type = "video/x-matroska"
        try:
            proc = await asyncio.create_subprocess_exec(
                *mkvmerge_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            log.warning("[MUX] mkvmerge not found; falling back to ffmpeg")
            use_mkvmerge_pipeline = False
        else:
            stderr_task = asyncio.create_task(_pump_stderr("mkvmerge", proc.stderr))
    if not use_mkvmerge_pipeline:
        ff_cmd = [FFMPEG_BIN, "-nostdin", "-hide_banner", "-loglevel", _ff_loglvl()]
        ff_cmd += ["-analyzeduration", "30M", "-probesize", "20M"]

        common_hdr = http_headers

        if start_seconds:
            ff_cmd += ["-ss", f"{start_seconds}"]
        if not use_local_file and common_hdr:
            ff_cmd += ["-headers", common_hdr]
        ff_cmd += ["-thread_queue_size", "4096", "-i", local_video_path if use_local_file else main_url]

        if start_seconds:
            ff_cmd += ["-ss", f"{start_seconds}"]
        if AUDIO_OFFSET_MS:
            ff_cmd += ["-itsoffset", f"{AUDIO_OFFSET_MS/1000.0}"]
        ff_cmd += ["-thread_queue_size", "4096", "-i", sidecar]

        ff_cmd += ["-map", "0"]
        ff_cmd += ["-map", "-0:a"]
        ff_cmd += ["-map", "1:a:0"]

        if KEEP_ORIGINAL_TRUEHD or drop_primary is False:
            audio_keep_infos = audio_streams
        else:
            audio_keep_infos = [info for info in audio_streams if info is not primary_audio]

        if audio_keep_infos:
            desc_parts: list[str] = []
            for info in audio_keep_infos:
                label = info.get("map_display") or info["map_spec"]
                detail = detail_map.get(info["order"]) if detail_map else None
                lang = (detail.get("language") if detail else info["stream"].get("Language")) or "und"
                lang = lang.lower()
                codec = detail.get("codec_name") if detail else (info["stream"].get("Codec") or "")
                flags = []
                if detail and detail.get("is_truehd"):
                    flags.append("truehd")
                if detail and detail.get("is_commentary"):
                    flags.append("commentary")
                flag_str = f" [{' '.join(flags)}]" if flags else ""
                desc_parts.append(f"{label}:{lang}{flag_str}:{codec}")
                ff_cmd += ["-map", f"0:a:{info['order']}?"]
            log.info("[MUX] keeping source audio map(s): %s", ", ".join(desc_parts))

        ff_cmd += ["-disposition:a:0", "default"]
        for out_offset in range(1, 1 + len(audio_keep_infos)):
            ff_cmd += [f"-disposition:a:{out_offset}", "0"]

        if sidecar_language:
            ff_cmd += ["-metadata:s:a:0", f"language={sidecar_language}"]
            log.info("[MUX] tagging sidecar audio language as '%s'", sidecar_language)

        if MUX_CONTAINER == "mp4":
            ff_cmd += [
                "-c", "copy",
                "-fflags", "+genpts",
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                "-f", "mp4",
                "pipe:1",
            ]
            content_type = "video/mp4"
        else:
            ff_cmd += [
                "-c", "copy",
                "-fflags", "+genpts",
                "-max_interleave_delta", "0",
                "-muxpreload", "0",
                "-muxdelay", "0",
                "-flush_packets", "1",
                "-f", "matroska",
                "pipe:1",
            ]
            content_type = "video/x-matroska"

        proc = await asyncio.create_subprocess_exec(
            *ff_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stderr_task = asyncio.create_task(_pump_stderr("ffmpeg", proc.stderr))
    else:
        # using mkvmerge
        stderr_task = asyncio.create_task(_pump_stderr("mkvmerge", proc.stderr))

    sidecar_ext = os.path.splitext(sidecar)[1].lstrip(".").lower() or "unknown"
    pipeline_mode = "mkvmerge" if use_mkvmerge_pipeline else "ffmpeg"
    log.info(
        "[MUX] remux start: sidecar='%s' ext=%s start=%.3fs keep_src=%s container=%s range_req='%s' est=%s reliable=%s source=%s mode=%s",
        os.path.basename(sidecar),
        sidecar_ext,
        start_seconds,
        KEEP_ORIGINAL_TRUEHD,
        MUX_CONTAINER,
        req.headers.get("Range"),
        _format_bytes(estimated_size),
        size_is_reliable,
        "local" if use_local_file else "http",
        pipeline_mode,
    )

    proc = await asyncio.create_subprocess_exec(
        *ff_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Async stderr pump for live diagnostics
    stderr_task = asyncio.create_task(_pump_stderr("ffmpeg", proc.stderr))

    # Build response headers
    filename = (meta.get("Name") or f"{item_id}") + (".mp4" if MUX_CONTAINER == "mp4" else ".mkv")
    headers = {
        "Content-Type": content_type,
        "Content-Disposition": f'inline; filename="{filename}"',
        "Cache-Control": "no-store",
    }

    # If client sent a Range header, some clients want 206 even for live streams.
    range_header = req.headers.get("Range")
    allow_partial = (
        FORCE_PARTIAL_206
        and effective_size
        and range_header
        and _is_initial_byte_range(range_header)
        and (size_is_reliable or FORCE_APPROX_RANGE)
    )

    if allow_partial:
        status_code = 206
        headers["Content-Range"] = f"bytes 0-{effective_size - 1}/{effective_size}"
        headers["Content-Length"] = str(effective_size)
        if not size_is_reliable and FORCE_APPROX_RANGE:
            log.info(
                "[MUX] Range requested; using approximate size due to FORCE_APPROX_RANGE est=%s",
                _format_bytes(estimated_size),
            )
        if fallback_cl_used:
            log.info(
                "[MUX] Content-Length fallback applied range=0-%d/%d",
                effective_size - 1,
                effective_size,
            )
    else:
        status_code = 200
        if range_header and FORCE_PARTIAL_206:
            if not estimated_size:
                log.info("[MUX] Range requested but mux size unknown; replying 200")
            elif not size_is_reliable:
                log.info("[MUX] Range requested but mux size estimate unsafe; replying 200")
        if range_header:
            headers.setdefault("Accept-Ranges", "none")

    async def body_iter():
        t0 = time.monotonic()
        total = 0
        first = True
        client_cancelled = False
        sample_bytes: Optional[bytearray] = bytearray() if MUX_HEADER_CAPTURE_LIMIT else None
        sample_path: Optional[str] = None
        sample_sha256: Optional[str] = None
        stream_error: Optional[str] = None
        last_progress_log = time.monotonic()
        last_activity = time.monotonic()
        idle_timeout_triggered = False
        max_duration_triggered = False
        start_time = time.monotonic()
        preroll_target = MUX_PREROLL_BYTES if MUX_PREROLL_BYTES > 0 else 0
        preroll_buffer = bytearray()
        preroll_flushed = preroll_target == 0
        try:
            assert proc.stdout is not None
            while True:
                if (
                    MUX_MAX_DURATION
                    and (time.monotonic() - start_time) >= MUX_MAX_DURATION
                ):
                    max_duration_triggered = True
                    log.info(
                        "[MUX] max duration reached secs=%.1f bytes=%d",
                        time.monotonic() - start_time,
                        total,
                    )
                    break
                chunk = await proc.stdout.read(64 * 1024)
                if not chunk:
                    # If ffmpeg exited, break; otherwise continue until it does.
                    if proc.returncode is not None:
                        if not preroll_flushed and preroll_buffer:
                            buffered = bytes(preroll_buffer)
                            preroll_buffer.clear()
                            preroll_flushed = True
                            total += len(buffered)
                            log.info(
                                "[MUX] preroll flush bytes=%d target=%d (on stream end)",
                                len(buffered),
                                preroll_target,
                            )
                            yield buffered
                        break
                    if (
                        MUX_IDLE_TIMEOUT
                        and (time.monotonic() - last_activity) >= MUX_IDLE_TIMEOUT
                    ):
                        idle_timeout_triggered = True
                        log.info(
                            "[MUX] idle timeout reached secs=%.1f bytes=%d",
                            time.monotonic() - last_activity,
                            total,
                        )
                        break
                    await asyncio.sleep(0.01)
                    continue
                if first:
                    log.info("[MUX] TTFB=%.3fs first_chunk=%d bytes", time.monotonic() - t0, len(chunk))
                    first = False
                if sample_bytes is not None and len(sample_bytes) < MUX_HEADER_CAPTURE_LIMIT:
                    take = min(len(chunk), MUX_HEADER_CAPTURE_LIMIT - len(sample_bytes))
                    if take > 0:
                        sample_bytes.extend(chunk[:take])
                last_activity = time.monotonic()
                out_chunk: Optional[bytes]
                if preroll_flushed:
                    out_chunk = chunk
                else:
                    preroll_buffer.extend(chunk)
                    if preroll_target and len(preroll_buffer) >= preroll_target:
                        out_chunk = bytes(preroll_buffer)
                        preroll_buffer.clear()
                        preroll_flushed = True
                        log.info(
                            "[MUX] preroll flush bytes=%d target=%d",
                            len(out_chunk),
                            preroll_target,
                        )
                    else:
                        continue

                total += len(out_chunk)
                if MUX_PROGRESS_INTERVAL and (time.monotonic() - last_progress_log) >= MUX_PROGRESS_INTERVAL:
                    log.info("[MUX] progress bytes=%d", total)
                    last_progress_log = time.monotonic()
                yield out_chunk
        except asyncio.CancelledError:
            # client went away
            client_cancelled = True
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            stream_error = f"{type(exc).__name__}: {exc}"
            log.exception("[MUX] stream generator failed item=%s", item_id)
        finally:
            with contextlib.suppress(ProcessLookupError):
                if proc.returncode is None:
                    proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            if not stderr_task.done():
                stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task

            sample_len = len(sample_bytes) if sample_bytes is not None else 0
            if sample_len:
                sample_data = bytes(sample_bytes)
                sample_sha256 = hashlib.sha256(sample_data).hexdigest()
                sample_hex = _hex_prefix(sample_data)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                sample_name = f"{item_id[:12]}-{timestamp}.bin"
                sample_path = os.path.join(MUX_SAMPLE_DIR, sample_name)
                try:
                    os.makedirs(MUX_SAMPLE_DIR, exist_ok=True)
                    with open(sample_path, "wb") as fh:
                        fh.write(sample_data)
                except Exception:
                    log.exception("[MUX] failed to persist header sample item=%s path=%s", item_id, sample_path)
                    sample_path = None
                else:
                    log.info(
                        "[MUX] head sample bytes=%d sha256=%s hex64=%s path=%s",
                        sample_len,
                        sample_sha256,
                        sample_hex,
                        sample_path,
                    )
                    log.info("[MUX] mkvinfo hint: mkvinfo %s", sample_path)
                    log.info(
                        "[MUX] curl header hint: curl -s -o %s -H 'User-Agent: Infuse-Direct/8.2' -H 'Range: bytes=0-65535' \"http://<proxy>:9999/Videos/%s/stream?MediaSourceId=%s&Static=true\"",
                        sample_name,
                        item_id,
                        item_id,
                    )

            diff_note = ""
            if estimated_size:
                diff = total - estimated_size
                human_diff = _format_bytes(abs(diff)) if diff else "0B"
                diff_note = f" diff={diff:+d}B (~{human_diff})"
            error_note = f" error={stream_error}" if stream_error else ""
            log.info(
                "[MUX] stream finished bytes=%d rc=%s cancelled=%s idle_timeout=%s max_duration=%s%s%s",
                total,
                proc.returncode,
                client_cancelled,
                idle_timeout_triggered,
                max_duration_triggered,
                diff_note,
                error_note,
            )

            if TRIM_AGGRESSIVE_POST_STREAM:
                with contextlib.suppress(asyncio.CancelledError):
                    await _trim_memory("post_mux_stream")

    return StreamingResponse(body_iter(), status_code=status_code, headers=headers)

# ------------------------- specific routes -----------------------------
@app.get("/Items/{item_id}/PlaybackInfo")
async def items_playback_info(item_id: str, request: Request):
    return await proxy_small(request, f"/Items/{item_id}/PlaybackInfo")

@app.get("/MediaSegments/{item_id}")
async def media_segments(item_id: str, request: Request):
    return await proxy_small(request, f"/MediaSegments/{item_id}")

@app.get("/Items/{item_id}")
async def items(item_id: str, request: Request):
    return await proxy_small(request, f"/Items/{item_id}")

@app.get("/Videos/{item_id}/stream")
async def videos_stream(item_id: str, request: Request):
    mux_resp = await proxy_stream_mux_infuse(request, item_id)
    if mux_resp is not None:
        return mux_resp
    return await proxy_stream_direct(request, f"/Videos/{item_id}/stream")

@app.get("/healthz")
async def health():
    return {"ok": True}

# ----------------------- catch-all pass-through ------------------------
@app.api_route("/{full_path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def catch_all(full_path: str, request: Request):
    jf_path = "/" + full_path
    return await proxy_small(request, jf_path)

# --------------------------- local dev ---------------------------------
if __name__ == "__main__":
    import uvicorn
    log.info("Application startup complete.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "9999")))
