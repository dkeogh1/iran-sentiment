"""
Message Batches results and clean-up, shared by every batch collect
(stance_local's checks, relabel, topic-label).

  stream_or_download  stream a batch's results through the SDK; when the
                      stream breaks (httpx ReadError on this host,
                      2026-09-23 and 2026-10-05), download the batch's
                      results_url to a file instead and read that, so
                      --results-file is a choice, not a necessity
  download_results    that download: plain httpx with the API key headers,
                      streamed to <file>.part, resumed with a Range request
                      after a dropped connection, renamed when whole; the
                      body is asked for unencoded, so the bytes held are
                      the offset a Range request counts
  delete_batch        the management call that deletes a collected batch,
                      so private replies are not kept on the API for the
                      29 days results otherwise stay; it flushes what the
                      collect wrote to disk first (os.sync), since the
                      delete cannot be undone
  create_batch        the create call, sent once (no SDK retry, which has
                      no idempotency key and could make a second batch)
  guarded_create      create_batch behind a "submitting" record, for the
                      submits that keep a plain state file (relabel,
                      topic-label)

Nothing here runs unless a collect or a submit calls it: there is no network
call at import or in an estimate.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from config import settings

logger = logging.getLogger(__name__)

API_VERSION_FALLBACK = "2023-06-01"  # what the SDK sends when the client names none


def stream_errors() -> tuple[type[BaseException], ...]:
    """What a results stream cut mid-body raises: a transport error from
    httpx, the SDK's connection error, or a JSON line cut short."""
    import anthropic
    import httpx

    return (
        httpx.TransportError,
        httpx.DecodingError,
        anthropic.APIConnectionError,
        json.JSONDecodeError,
    )


def api_headers(client) -> dict[str, str]:
    """The headers the SDK authenticates with: the client's API key (or
    ANTHROPIC_API_KEY) and its anthropic-version; and accept-encoding
    identity (httpx asks for gzip by default), so what download_results
    holds on disk is the raw body a Range offset counts."""
    key = getattr(client, "api_key", None) or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (.env): can't download batch results")
    default = {
        str(k).lower(): v for k, v in (getattr(client, "default_headers", None) or {}).items()
    }
    return {
        "x-api-key": key,
        "anthropic-version": default.get("anthropic-version") or API_VERSION_FALLBACK,
        "accept": "application/binary",
        "accept-encoding": "identity",
    }


def _http():
    import httpx

    return httpx.Client(
        timeout=httpx.Timeout(60.0, read=settings.BATCH_RESULTS_READ_TIMEOUT_S),
        follow_redirects=False,
    )


def _results_url(client, batch_id: str) -> str:
    b = client.messages.batches.retrieve(batch_id)
    url = getattr(b, "results_url", None)
    if not url:
        raise RuntimeError(
            f"batch {batch_id} has no results_url (status "
            f"{getattr(b, 'processing_status', '?')}): has it ended?"
        )
    if not urlsplit(url).scheme:
        url = urljoin(str(getattr(client, "base_url", "https://api.anthropic.com/")), url)
    return url


class _EncodedResume(Exception):
    """A Range request answered 206 with an encoded body: its offset counts
    encoded bytes, what is held is decoded, so the two don't line up."""


def _encoded(r) -> bool:
    return (r.headers.get("content-encoding") or "identity").strip().lower() not in (
        "identity",
        "",
    )


def _get_once(http, url: str, headers: dict, part: Path, max_redirects: int = 3) -> None:
    """One GET of url into part, appending when it answers a Range request
    with 206, else rewriting it. A redirect to another host is followed
    without the API key. A body sent encoded despite accept-encoding
    identity is never resumed: an encoded 206 drops what is held and starts
    over, and an encoded body cut short leaves nothing held. Raises on a
    dropped connection or an error status."""
    try:
        _get(http, url, headers, part, max_redirects)
    except _EncodedResume:
        logger.warning(
            "batch results download: the resume came back encoded; dropping the %d bytes "
            "held and starting over",
            part.stat().st_size if part.exists() else 0,
        )
        part.unlink(missing_ok=True)
        _get(http, url, headers, part, max_redirects)


def _get(http, url: str, headers: dict, part: Path, max_redirects: int) -> None:
    have = part.stat().st_size if part.exists() else 0
    hdrs = {**headers, **({"range": f"bytes={have}-"} if have else {})}
    origin = urlsplit(url).netloc
    for _ in range(max_redirects + 1):
        with http.stream("GET", url, headers=hdrs) as r:
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                url = urljoin(url, r.headers["location"])
                if urlsplit(url).netloc != origin:
                    hdrs = {
                        k: v for k, v in hdrs.items() if k not in ("x-api-key", "anthropic-version")
                    }
                continue
            if r.status_code == 416 and have:  # nothing past what we hold: it is whole
                return
            r.raise_for_status()
            encoded = _encoded(r)
            if have and r.status_code == 206 and encoded:
                raise _EncodedResume()
            mode = "ab" if have and r.status_code == 206 else "wb"
            try:
                with open(part, mode) as f:
                    f.writelines(r.iter_bytes())
            except BaseException:
                if encoded:  # decoded bytes held: no Range offset can resume them
                    part.unlink(missing_ok=True)
                raise
            return
    raise RuntimeError(f"more than {max_redirects} redirects downloading batch results")


def download_results(
    client, batch_id: str, dest: Path, *, http=None, retries: int | None = None, sleep=None
) -> Path:
    """Download the ended batch's results JSONL to dest (temp file, then
    rename). A dropped connection or a 5xx is retried, resuming from what
    the temp file holds; a 4xx is not. The temp file is kept when every try
    fails, so the next collect resumes it."""
    import httpx

    retries = settings.BATCH_RESULTS_DOWNLOAD_RETRIES if retries is None else retries
    sleep = sleep or time.sleep
    url = _results_url(client, batch_id)
    headers = api_headers(client)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    own = http is None
    http = http or _http()
    try:
        for attempt in range(retries + 1):
            try:
                _get_once(http, url, headers, part)
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500 or attempt == retries:
                    raise
                err = f"HTTP {e.response.status_code}"
            except httpx.TransportError as e:
                if attempt == retries:
                    raise
                err = type(e).__name__
            except httpx.DecodingError as e:  # a body that will not decode: start it over
                part.unlink(missing_ok=True)
                if attempt == retries:
                    raise
                err = type(e).__name__
            wait = min(2**attempt, 60)
            held = part.stat().st_size if part.exists() else 0
            logger.warning(
                "batch %s results download: %s with %d bytes held; retry %d of %d in %ds",
                batch_id,
                err,
                held,
                attempt + 1,
                retries,
                wait,
            )
            sleep(wait)
    finally:
        if own:
            http.close()
    os.replace(part, dest)
    logger.info("batch %s results downloaded -> %s (%d bytes)", batch_id, dest, dest.stat().st_size)
    return dest


def stream_or_download(client, batch_id: str, dest: Path, reader) -> tuple[list, Path | None]:
    """(results, path): the SDK's results stream as a list and None, or,
    when the stream breaks, the results read by `reader` from results_url
    downloaded to dest, and dest."""
    try:
        return list(client.messages.batches.results(batch_id)), None
    except stream_errors() as e:
        logger.warning(
            "batch %s: the results stream broke (%s); downloading results_url to %s",
            batch_id,
            type(e).__name__,
            dest,
        )
    download_results(client, batch_id, dest)
    return list(reader(Path(dest))), Path(dest)


def create_batch(client, requests: list):
    """Create the batch with one attempt. The SDK retries a POST after a
    timeout, a dropped connection, a 408/409/429 or a 5xx, and sends no
    idempotency key, so a create the server accepted whose answer was lost
    would be sent again and make a second batch the caller never records:
    billed, outside every run cap, and (for a reply check) never deleted.
    With no retry, a failure without an answer leaves the caller's
    "submitting" record for a person to check against the Console. Stub
    clients without with_options are called as they are."""
    one = client.with_options(max_retries=0) if hasattr(client, "with_options") else client
    return one.messages.batches.create(requests=requests)


def create_refused(e: Exception) -> bool:
    """The API answered the one create call with a refusal, sent before any
    batch exists: a client error (4xx, 429 included) or 529 overloaded. A
    timeout, a dropped connection or another 5xx leaves that unknown."""
    import anthropic

    return isinstance(e, anthropic.APIStatusError) and (e.status_code < 500 or e.status_code == 529)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def pending_message(path: Path, st: dict) -> str:
    """What to do about a submit that stopped before it recorded a batch id."""
    return (
        f"the submit at {st.get('submitted_at')} of {st.get('n_submitted')} requests stopped "
        "before it recorded a batch id, so a new submit could pay again for them. If the "
        "Console's Batches page lists a batch created then with that many requests, set "
        f'"batch_id" to its id and "status" to "in_progress" in {path.name}, then collect; if '
        f'it lists none, replace {path.name} with its "previous" record (delete the file when '
        "that is null)"
    )


def guarded_create(client, requests: list, path: Path, pending: dict):
    """create_batch for a submit with a plain state file at `path`: writes
    `pending` there first as status "submitting" with no batch id (the
    state it replaces under "previous"), so a create that dies without an
    answer leaves a record the next submit refuses on (pending_message). A
    refusal (create_refused: no batch was made) puts the previous state
    back. Returns the batch; the caller writes the new state."""
    prev = json.loads(path.read_text()) if path.exists() else None
    rec = {**pending, "batch_id": None, "status": "submitting", "previous": prev}
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, rec)
    try:
        return create_batch(client, requests)
    except Exception as e:
        if create_refused(e):
            if prev is None:
                path.unlink(missing_ok=True)
            else:
                _write_json(path, prev)
        else:
            logger.error(
                "batch create failed without an answer (%s): %s",
                type(e).__name__,
                pending_message(path, rec),
            )
        raise


def delete_batch(client, batch_id: str) -> dict:
    """Delete the batch (and the results the API keeps for 29 days). Returns
    {"status": "deleted" | "failed", "at": ..., "error": ...}; a failure is
    logged, not raised: the labels are already saved. A batch the API no
    longer has (404: deleted in the Console, say) counts as deleted. The
    delete cannot be undone, so everything written so far (the collect's
    labels and results archive, new files whose blocks ext4 may still hold
    in memory) is flushed to disk first: a power-off just after the delete
    must not leave the labels empty with the results gone from the API."""
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.sync()
    try:
        client.messages.batches.delete(batch_id)
    except Exception as e:  # noqa: BLE001 -- the collect stands; the state records the failure
        if getattr(e, "status_code", None) == 404:
            logger.info("batch %s: not on the API any more (deleted already)", batch_id)
            return {"status": "deleted", "at": at, "note": "not found: deleted already"}
        logger.warning(
            "batch %s: delete failed (%s: %s); its results stay on the API until it "
            "is deleted (Console, or collect again)",
            batch_id,
            type(e).__name__,
            str(e)[:200],
        )
        return {"status": "failed", "at": at, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    logger.info("batch %s: deleted on the API (its requests and results)", batch_id)
    return {"status": "deleted", "at": at}
