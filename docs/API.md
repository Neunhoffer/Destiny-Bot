# Destiny Bot local JSON API

`destiny serve` starts the server at `http://127.0.0.1:8765/`. The browser UI is
the only intended client. All JSON. Errors use FastAPI's shape, `{"detail": "<message the UI can show as is>"}`.

## Security (local-only server)

- The server binds to `127.0.0.1`. Requests whose `Host` header isn't
  `127.0.0.1:<port>` or `localhost:<port>` get `403`. This blocks DNS rebinding.
- `GET /` serves `index.html` with a per-process random token in
  `<meta name="destiny-token" content="...">`.
- Every `/api/*` request must send the header `X-Destiny-Token: <token>`.
  Otherwise the server returns `401`. The one exception is
  `GET /api/jobs/{id}/files/{name}`, which may pass `?token=` so a plain `<a href>`
  download works.
- For requests with an `Origin` header, the origin must be the server's own origin.
  Otherwise the server returns `403`.
- Uploads are capped at 300 MB; larger ones get `413`.
- The API key is never returned. Settings report `anthropic_api_key_set: true/false`.

## Types

```jsonc
// BookSummary
{"id": "3fa2...", "title": "Campbell Biology", "author": "Urry", "source_name": "bio.pdf",
 "source_type": "pdf", "added_at": "2026-10-14T02:00:00Z", "section_count": 42, "page_count": 1200,
 "progress": {"section_id": "s3", "utterance": 17, "updated_at": "..."} | null}

// TocEntry
{"id": "s3", "title": "Chapter 2: The Chemistry of Life", "level": 1, "page_start": 28, "page_end": 45,
 "block_count": 120}

// Block (as displayed)
{"kind": "heading|paragraph|list_item|quote|caption|figure|table|equation|footnote|code",
 "text": "...", "level": 1 | null, "rows": [["a","b"]] | null, "page": 28 | null}

// Utterance (see models.Utterance)
{"index": 0, "block": 0, "start": 0, "end": 42, "kind": "paragraph", "speech": "...", "pause_after_ms": 350}

// Engine (in /api/status)
{"name": "piper", "label": "Piper (natural, offline)", "online": false, "available": true,
 "reason": "", "default_voice": "en_US-lessac-medium"}

// Voice
{"id": "en_US-lessac-medium", "name": "Lessac (US English)", "engine": "piper", "language": "en-US",
 "gender": null, "online": false, "installed": true, "size_mb": 63.2}

// Job
{"id": "j_1a2b", "kind": "import|export|voice_download", "status": "running|done|error|cancelled",
 "progress": 0.42, "message": "Reading page 120 of 300", "error": null | "friendly message",
 "result": null | {...kind specific...}}
```

## Endpoints

| Method & path | Body / query | Response |
|---|---|---|
| `GET /` | | the UI (`index.html` with token) |
| `GET /static/{file}` | | UI assets from `destiny_bot/web/` |
| `GET /api/status` | | `{"version", "engines": [Engine], "resolved_engine": "piper", "ai": {"enabled", "configured", "model"}, "ocr": {"available", "reason"}, "data_dir": "/path"}` |
| `GET /api/settings` | | `Settings.public_dict()` |
| `PUT /api/settings` | any subset of settings fields. `reading` and `voices` may be partial. `anthropic_api_key: "sk-..."` sets the key and `""` clears it | `Settings.public_dict()`; `422` on invalid values |
| `GET /api/library` | | `{"books": [BookSummary]}` |
| `POST /api/library/import` | multipart field `file` | `{"job_id"}`. The job result is `{"book": BookSummary, "warnings": [str]}`. Unsupported type gives `400` immediately |
| `POST /api/library/paste` | `{"title", "text"}` | `{"book": BookSummary}` |
| `DELETE /api/library/{id}` | | `{"ok": true}` |
| `GET /api/books/{id}` | | `{"book": BookSummary, "toc": [TocEntry], "warnings": [str]}`; marks the book as opened |
| `GET /api/books/{id}/sections/{sid}` | | `{"section": TocEntry, "blocks": [Block], "utterances": [Utterance], "prev_id": "s2" or null, "next_id": "s4" or null}`. Utterances come from `plan_section` with the saved `reading` options |
| `GET /api/books/{id}/progress` | | `{"progress": Progress or null}` |
| `PUT /api/books/{id}/progress` | `{"section_id", "utterance"}` | `{"progress": Progress}` |
| `GET /api/books/{id}/bookmarks` | | `{"bookmarks": [Bookmark]}` |
| `POST /api/books/{id}/bookmarks` | `{"section_id", "utterance", "label"}` | `{"bookmark": Bookmark}` |
| `DELETE /api/books/{id}/bookmarks/{bid}` | | `{"ok": true}` |
| `GET /api/books/{id}/pages/{n}.png` | | PNG render of PDF page n, 1-based (`404` if not a PDF) |
| `GET /api/voices` | `?engine=piper` | `{"voices": [Voice]}`, installed voices only |
| `GET /api/voices/catalog` | | `{"voices": [CatalogVoice + {"installed": bool}]}` (Piper downloads) |
| `POST /api/voices/download` | `{"voice_id"}` | `{"job_id"}`; the job result is `{"voice_id"}` |
| `POST /api/tts` | `{"text", "engine"?, "voice"?, "rate"?}`. `rate` is 0.5–3.0 (default 1.0). The voice engine itself produces the speed, which sounds more natural than speeding up in the browser | audio bytes, `Content-Type: audio/wav` or `audio/mpeg`. Cached. `400` for an empty text or the browser engine, `403` for an online engine while `online_tts_allowed` is false, `503` if the engine is unavailable. `engine` defaults to the resolved engine and `voice` to `settings.voices[engine]`. Leading and trailing silence is trimmed from WAV output, so the UI controls the pauses through `pause_after_ms` |
| `POST /api/ai/explain` | `{"book_id", "section_id", "passage"}` | `{"text"}` |
| `POST /api/ai/ask` | `{"book_id", "section_id", "question", "history": [{"role": "user" or "assistant", "content"}]}` | `{"text"}` |
| `POST /api/ai/summarize` | `{"book_id", "section_id", "length": "short" or "detailed"}` | `{"text"}` |
| `POST /api/ai/quiz` | `{"book_id", "section_id", "count": 5}` | `{"questions": [{"question", "choices", "answer_index", "explanation"}]}` |
| `POST /api/ai/narrate` | `{"book_id", "section_id", "block": 12}` | `{"text"}`: a spoken explanation of a table or equation block |
| `POST /api/ai/describe-page` | `{"book_id", "page"}` | `{"text"}`: describes figures and diagrams on a PDF page |
| `POST /api/export` | `{"book_id", "section_ids": ["s1"] or null, "engine"?, "voice"?, "format": "mp3" or "wav", "rate": 1.0}` | `{"job_id"}`. The job result is `{"files": [{"name", "url", "size"}]}` |
| `GET /api/jobs/{id}` | | Job |
| `POST /api/jobs/{id}/cancel` | | Job |
| `GET /api/jobs/{id}/files/{name}` | `?token=` allowed | the exported file (attachment) |
| `POST /api/wipe` | `{"confirm": "DELETE"}` | `{"ok": true}`. Deletes the library, audio cache, exports and settings (the API key too). Installed voices are kept |

AI endpoints return `409` with a friendly `detail` when AI is turned off or no key
is set, and `502` with a friendly `detail` when the Anthropic call fails.
Section context sent to Claude is `section_plain_text(section, max_chars=150_000)`.

## Details and edge cases

- Each block in a section response also has `"notes": true|false`, which is true for PowerPoint speaker notes.
- `422` responses carry `detail` as a plain string such as `"Invalid request: rate: ..."`, not FastAPI's list form.
- `GET /api/voices` and `status.engines[].default_voice` leave out online engines while `online_tts_allowed` is false.
- `POST /api/voices/download` accepts any well-formed Piper voice id; a malformed id gets `400`.
- `/api/tts` returns `400` for text over 5,000 characters, and `503` when the engine has no voice at all. A requested or saved voice that isn't installed falls back to the engine's default voice.
- `PUT /api/settings` returns `422` for an unknown `tts_engine`.
- Every response carries `X-Frame-Options: DENY` and `Cross-Origin-Resource-Policy: same-origin`. `/static` responses use `Cache-Control: no-cache`, and JS/CSS media types are set explicitly, because Windows registries often map `.js` to `text/plain`.
- A job waiting for a free worker reports `"status": "running"` with a "Waiting…" message.
