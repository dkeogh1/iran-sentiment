# Truth Social API notes

How the collectors reach Truth Social, a Mastodon fork behind Cloudflare bot
mitigation that fingerprints the TLS handshake. Observed September 2026; the
code in `src/collectors/truthsocial_collector.py` and the `TS_*` settings in
`config/settings.py` are the source of truth.

## Transport

- Plain httpx, requests and curl get 403 whatever the headers or token.
- `curl_cffi` with `impersonate="chrome"` gets through. It comes with
  truthbrush (`pip install -e '.[truthsocial]'`).
- truthbrush 0.2.x handles authenticated requests and rate-limit backoff. Only
  `pull_statuses` still goes through it; the reply walker is ours.

## Endpoints

| Endpoint | Auth | Notes |
|---|---|---|
| `/api/v1/accounts/lookup` | no | |
| `/api/v1/accounts/:id/statuses` | no | pages backward with `max_id` only |
| `/api/v1/statuses/:id` | no | single post |
| `/api/v1/statuses/:id/context` | yes | Cloudflare blocks it even with auth: don't use |
| `/api/v1/statuses/:id/context/descendants` | yes | dead (bare `404 page not found`); truthbrush 0.2.5 still calls it |
| `/api/v2/statuses/:id/context/descendants` | yes | the one that works: 20 per page, `?sort=oldest`, Link `rel="next"` with `offset=` (`iter_descendants`, `TS_DESCENDANTS_PATH`) |

## Pagination and rate limits

- `/accounts/:id/statuses` ignores `min_id` and returns the newest page, so
  there is no forward walk. Incremental refresh walks backward to the cache
  edge and holds pages in `<handle>.partial.jsonl` until it reaches it.
- `limit=40` is honoured as 20 per page.
- Anonymous: about 5 pages, then HTTP 429 with no Retry-After or rate-limit
  headers; a 60 s wait clears it (~100 posts a minute sustained).
  `TS_PAGE_DELAY_S` and the backoff settings pace it.
- Authenticated: 300 requests per 5 minutes (`x-ratelimit-limit`); the
  reply walker paces at 1 request a second (`TS_AUTH_PAGE_DELAY_S`) and truthbrush
  sleeps when fewer than 50 remain. A 16k-reply post takes about 15 minutes.

## Login

- Token endpoint `POST /oauth/v2/token` with a JSON body (not
  `/oauth/token`, not form-encoded), using the web app's public client
  ID/secret (`TRUTH_SOCIAL_WEB_CLIENT_*`). Auth calls also send the web
  app's `Browser` / `OS` headers (`TS_AUTH_HEADERS`).
- A new device gets 403 `security_code_required` with a `challenge_id`. The
  three-step flow (choose delivery method, then `verify_security_code`) is in
  the comment above `TS_SECURITY_CODE_DELIVERY_ENDPOINT`. Each token request
  mints a new challenge, so the later steps must use the `challenge_id` from
  the same run.
- `python -m src.cli ts-login` drives it (`--deliver email|sms`, then
  `--challenge-id X --code N`) and writes `TRUTHSOCIAL_TOKEN` to `.env`; the
  user then re-encrypts `secrets.env`. The reply walker and truthbrush's
  `Api(token=)` reuse the token. Without one, `collect-replies` and
  `probe-auth` fail; the anonymous statuses walk still works.
- Env names: truthbrush reads `TRUTHSOCIAL_USERNAME` / `TRUTHSOCIAL_PASSWORD`
  (no underscore between TRUTH and SOCIAL); `probe-auth` reads
  `TRUTH_SOCIAL_*`. Set both.
