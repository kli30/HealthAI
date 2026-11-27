"""Fetch YouTube captions and description without importing the RAG stack."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit
from xml.etree.ElementTree import ParseError

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")


def normalize_url(url: object) -> tuple[str, str] | None:
    """Validate an HTTP(S) YouTube URL; never pass user URLs to providers."""
    if not isinstance(url, str) or any(c.isspace() or ord(c) < 32 for c in url):
        return None
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or parts.username or parts.password:
            return None
        if parts.port is not None or "%" in parts.path or "%" in parts.query:
            return None
        if parts.netloc.lower() not in (
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "www.youtu.be",
        ):
            return None
        values = parse_qs(parts.query, keep_blank_values=True).get("v", [])
        if len(values) > 1:
            return None
        if parts.hostname in ("youtu.be", "www.youtu.be"):
            video_id = parts.path[1:]
        elif parts.path == "/watch":
            if len(values) != 1:
                return None
            video_id = values[0]
        else:
            match = re.fullmatch(r"/(?:shorts|embed|live)/([^/]+)", parts.path)
            if not match:
                return None
            video_id = match[1]
        if not _VIDEO_ID.fullmatch(video_id):
            return None
        return video_id, f"https://www.youtube.com/watch?v={video_id}"
    except ValueError:
        return None


class ProviderError(Exception):
    """A provider's classified failure (never a raw network error in results)."""

    def __init__(self, status: str):
        super().__init__(status)
        self.status = status


class TextProvider(Protocol):
    def transcript(self, video_id: str) -> str: ...
    def metadata(self, video_id: str) -> dict[str, Any]: ...


class DefaultYouTubeProvider:
    """Caption-first adapters; optional packages are imported only on use."""

    @staticmethod
    def _validate_id(video_id: str) -> None:
        if not _VIDEO_ID.fullmatch(video_id):
            raise ValueError("Expected an 11-character YouTube video ID")

    @staticmethod
    def _is_block_signal(message: str) -> bool:
        normalized = " ".join(message.casefold().replace("’", "'").split())
        return any(
            token in normalized
            for token in (
                "http error 403",
                "http error 429",
                "too many requests",
                "confirm you're not a bot",
                "confirm you are not a bot",
                "your ip is blocked",
                "your ip has been blocked",
                "your ip is likely being blocked by youtube",
            )
        )

    @staticmethod
    def _validate_player_response(data: Any) -> None:
        """Validate the JSON consumed by youtube-transcript-api's v1 parser.

        Missing captions establish absence only after an explicit OK status.
        Non-OK statuses are left to the dependency's typed error classifier.
        Present-but-malformed containers/fields are never evidence of absence.
        """
        if not isinstance(data, dict):
            raise ProviderError("provider_error")
        playability = data.get("playabilityStatus")
        if not isinstance(playability, dict) or playability.get("status") not in (
            "OK",
            "ERROR",
            "LOGIN_REQUIRED",
            "UNPLAYABLE",
            "LIVE_STREAM_OFFLINE",
            "CONTENT_CHECK_REQUIRED",
            "AGE_CHECK_REQUIRED",
        ):
            raise ProviderError("provider_error")
        if playability["status"] != "OK":
            if (
                not isinstance(playability.get("reason"), str)
                or not playability["reason"]
            ):
                raise ProviderError("provider_error")
            if DefaultYouTubeProvider._is_block_signal(playability["reason"]):
                raise ProviderError("blocked")
            return
        captions = data.get("captions", {})
        if not isinstance(captions, dict):
            raise ProviderError("provider_error")
        if "playerCaptionsTracklistRenderer" not in captions:
            return
        renderer = captions["playerCaptionsTracklistRenderer"]
        if not isinstance(renderer, dict):
            raise ProviderError("provider_error")

        def validate_language(entry: Any, name_key: str) -> None:
            # Validate the exact fields the v1 listing parser consumes, including
            # scalar types which it otherwise accepts and fails on only later.
            try:
                code = entry["languageCode"]
                runs = entry[name_key]["runs"]
                text = runs[0]["text"]
            except (KeyError, TypeError, IndexError) as exc:
                raise ProviderError("provider_error") from exc
            if (
                not isinstance(code, str)
                or not code
                or not isinstance(runs, list)
                or not isinstance(text, str)
                or not text
            ):
                raise ProviderError("provider_error")

        for key, name_key in (
            ("captionTracks", "name"),
            ("translationLanguages", "languageName"),
        ):
            entries = renderer.get(key, [])
            if not isinstance(entries, list):
                raise ProviderError("provider_error")
            for entry in entries:
                validate_language(entry, name_key)
                if key == "captionTracks" and (
                    not isinstance(entry.get("baseUrl"), str) or not entry["baseUrl"]
                ):
                    raise ProviderError("provider_error")

    def transcript(self, video_id: str) -> str:
        self._validate_id(video_id)
        try:
            import requests
            from youtube_transcript_api import (
                AgeRestricted,
                CouldNotRetrieveTranscript,
                NoTranscriptFound,
                RequestBlocked,
                TranscriptsDisabled,
                VideoUnavailable,
                VideoUnplayable,
                YouTubeRequestFailed,
                YouTubeTranscriptApi,
            )
        except ImportError as exc:
            raise ProviderError("dependency_missing") from exc

        validate_player_response = self._validate_player_response

        class TimeoutSession(requests.Session):
            def post(self, url: str | bytes, *args: Any, **kwargs: Any) -> Any:
                response = super().post(url, *args, **kwargs)
                # This session belongs solely to the transcript API. Validate at
                # its supported HTTP-client boundary, before .list() discards the
                # player status. Do not inspect error bodies: upstream preserves
                # HTTP block/transient classifications via raise_for_status().
                parts = urlsplit(url)
                if (
                    parts.hostname == "www.youtube.com"
                    and parts.path == "/youtubei/v1/player"
                    and 200 <= response.status_code < 300
                ):
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise ProviderError("provider_error") from exc
                    validate_player_response(data)
                return response

            def request(
                self, method: str, url: str | bytes, *args: Any, **kwargs: Any
            ) -> Any:
                kwargs["timeout"] = 10
                return super().request(method, url, *args, **kwargs)

        try:
            with TimeoutSession() as session:
                session.trust_env = False
                try:
                    tracks = list(
                        YouTubeTranscriptApi(http_client=session).list(video_id)
                    )
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    IndexError,
                    ValueError,
                ) as exc:
                    # The dependency's listing parser also consumes untrusted
                    # JSON without wrapping all schema errors. Scope this guard
                    # to the default adapter, not arbitrary injected providers.
                    raise ProviderError("provider_error") from exc
                if not tracks:
                    raise ProviderError("no_transcript")
                # Prefer English, then manual captions; fall back to any language.
                tracks.sort(
                    key=lambda t: (
                        t.language_code.casefold().split("-")[0] != "en",
                        t.is_generated,
                    )
                )
                try:
                    fetched = tracks[0].fetch()
                except (ParseError, ValueError, KeyError) as exc:
                    # The upstream XML parser/float timing conversion can fail
                    # without wrapping malformed caption data in its own errors.
                    raise ProviderError("provider_error") from exc
                return " ".join(segment.text for segment in fetched)
        except RequestBlocked as exc:
            raise ProviderError("blocked") from exc
        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            raise ProviderError("no_transcript") from exc
        except (VideoUnavailable, VideoUnplayable, AgeRestricted) as exc:
            raise ProviderError("unavailable") from exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise ProviderError("transient_error") from exc
        except YouTubeRequestFailed as exc:
            # v1.x preserves HTTPError via implicit exception context, not a field.
            response = getattr(exc.__cause__ or exc.__context__, "response", None)
            code = getattr(response, "status_code", None)
            status = (
                "blocked"
                if code in (403, 429)
                else "transient_error"
                if code and code >= 500
                else "provider_error"
            )
            raise ProviderError(status) from exc
        except (CouldNotRetrieveTranscript, requests.RequestException) as exc:
            raise ProviderError("provider_error") from exc

    def metadata(self, video_id: str) -> dict[str, Any]:
        self._validate_id(video_id)
        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.utils import DownloadError
        except ImportError as exc:
            raise ProviderError("dependency_missing") from exc

        is_block_signal = self._is_block_signal

        class QuietLogger:
            def debug(self, message: str) -> None:
                pass

            def warning(self, message: str) -> None:
                if is_block_signal(message):
                    # Abort intermediate client fallback, not just terminal errors.
                    raise ProviderError("blocked")

            def error(self, message: str) -> None:
                if is_block_signal(message):
                    raise ProviderError("blocked")

        options = {
            "skip_download": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "logger": QuietLogger(),
            "retries": 0,
            "extractor_retries": 0,
            "fragment_retries": 0,
            "socket_timeout": 10,
            "ignore_no_formats_error": True,
            "writesubtitles": False,
            "writeautomaticsub": False,
        }
        try:
            with YoutubeDL(options) as extractor:
                info = extractor.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False
                )
        except DownloadError as exc:
            # yt-dlp wraps extractor/HTTP errors in DownloadError. Be conservative:
            # unrecognized failures must never masquerade as missing captions.
            message = str(exc).lower()
            if is_block_signal(message):
                status = "blocked"
            elif any(
                token in message
                for token in (
                    "private video",
                    "video unavailable",
                    "video has been removed",
                    "sign in to confirm your age",
                )
            ):
                status = "unavailable"
            elif any(
                token in message
                for token in (
                    "timed out",
                    "timeout",
                    "temporary failure",
                    "http error 500",
                    "http error 502",
                    "http error 503",
                    "connection",
                )
            ):
                status = "transient_error"
            else:
                status = "provider_error"
            raise ProviderError(status) from exc
        if not isinstance(info, dict):
            raise ProviderError("provider_error")
        return {
            "author": info.get("channel") or info.get("uploader") or "",
            "title": info.get("title") or "",
            "description": info.get("description") or "",
            "keywords": info.get("tags") or [],
        }


class YouTubeTextClient:
    """Reusable synchronous client. Cache/cooldown are local to this instance.

    Providers raise ProviderError; unexpected programming errors propagate.
    No automatic retries, downloads, credentials or paid services are used.
    Use a separate instance per worker; this client is not thread-safe.
    """

    def __init__(
        self,
        provider: TextProvider | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        cache_size: int = 128,
    ):
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.provider = provider if provider is not None else DefaultYouTubeProvider()
        self.clock = clock
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict[str, tuple[float, str, Any]]] = OrderedDict()
        self._blocked_until = 0.0

    def _stage(self, video_id: str, stage: str) -> tuple[str, Any]:
        now = self.clock()
        entry = self._cache.get(video_id, {})
        cached = entry.get(stage)
        if cached and cached[0] > now:
            self._cache.move_to_end(video_id)
            return cached[1], cached[2]
        if now < self._blocked_until:
            return "blocked", None
        self._cache[video_id] = entry
        self._cache.move_to_end(video_id)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        try:
            value = getattr(self.provider, stage)(video_id)
            if stage == "transcript" and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ProviderError("provider_error")
            if stage == "metadata" and not isinstance(value, dict):
                raise ProviderError("provider_error")
            status = "ok"
        except ProviderError as exc:
            status, value = exc.status, None
        # Expiry starts when the outcome is observed, not before network work.
        now = self.clock()
        ttl = 3600 if status in ("ok", "no_transcript", "unavailable") else 30
        if status == "blocked":
            ttl = 300
            self._blocked_until = now + ttl
        entry[stage] = (now + ttl, status, deepcopy(value))
        return status, value

    def get(self, url: object) -> dict[str, Any]:
        """Return stable required fields plus metadata and per-stage statuses."""
        result: dict[str, Any] = {
            "video_id": "",
            "author": "",
            "url": url if isinstance(url, str) else "",
            "transcript": "",
            "status": "invalid_url",
            "title": "",
            "description": "",
            "keywords": [],
            "transcript_status": "not_requested",
            "metadata_status": "not_requested",
        }
        normalized = normalize_url(url)
        if normalized is None:
            return result
        video_id, canonical = normalized
        result.update(video_id=video_id, url=canonical)
        transcript_status, transcript = self._stage(video_id, "transcript")
        metadata_status, metadata = self._stage(video_id, "metadata")
        result.update(
            transcript_status=transcript_status, metadata_status=metadata_status
        )
        if transcript_status == "ok":
            result["transcript"] = transcript
        if metadata_status == "ok":
            for key in ("author", "title", "description"):
                value = metadata.get(key)
                result[key] = value if isinstance(value, str) else ""
            keywords = metadata.get("keywords")
            if isinstance(keywords, list):
                result["keywords"] = [k for k in keywords if isinstance(k, str)]
        result["status"] = (
            ("ok" if metadata_status == "ok" else "partial")
            if transcript_status == "ok"
            else transcript_status
        )
        return deepcopy(result)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get YouTube captions and video description as JSON"
    )
    parser.add_argument("url", help="HTTP(S) YouTube video URL")
    args = parser.parse_args()
    result = YouTubeTextClient().get(args.url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
