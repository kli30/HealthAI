"""Offline contract tests: no YouTube or LLM requests."""

import io
import json
import unittest
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from yt_dlp import YoutubeDL

from src.youtube_text import (
    DefaultYouTubeProvider,
    ProviderError,
    YouTubeTextClient,
    normalize_url,
)


@contextmanager
def caption_payload(payload):
    """Keep real caption HTTP-response parsing; replace only network/listing."""
    import requests
    from youtube_transcript_api._transcripts import Transcript

    response = requests.Response()
    response.status_code = 200
    response._content = payload.encode()
    with requests.Session() as session:
        track = Transcript(
            session,
            "abcdefgh_-1",
            "https://www.youtube.com/api/timedtext",
            "English",
            "en",
            False,
            [],
        )
        with (
            patch("requests.Session.get", return_value=response) as request,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as api,
        ):
            api.return_value.list.return_value = [track]
            yield request


@contextmanager
def player_payload(payload, *, caption_xml=None, status_code=200):
    """Exercise the real listing/parser with only HTTP transport replaced."""
    import requests

    watch = requests.Response()
    watch.status_code = 200
    watch._content = b'"INNERTUBE_API_KEY":"offline"'
    player = requests.Response()
    player.status_code = status_code
    player._content = (
        payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    )
    caption = requests.Response()
    caption.status_code = 200
    caption._content = (caption_xml or "").encode()
    with (
        patch(
            "requests.Session.get",
            side_effect=lambda url, **kwargs: watch if "/watch?" in url else caption,
        ) as get,
        patch("requests.Session.post", return_value=player) as post,
    ):
        yield get, post


class ProviderTests(unittest.TestCase):
    def test_real_malformed_listing_keeps_metadata_and_short_cache(self):
        good_track = {
            "baseUrl": "https://www.youtube.com/api/timedtext",
            "languageCode": "en",
            "name": {"runs": [{"text": "English"}]},
        }
        renderers = [
            None,
            [],
            "bad",
            {"captionTracks": None},
            {"captionTracks": {}},
            {"captionTracks": [None]},
            {"captionTracks": [{}]},
        ]
        for key, value in (
            ("baseUrl", None),
            ("languageCode", []),
            ("name", None),
            ("name", {"runs": []}),
            ("name", {"runs": [{"text": 1}]}),
        ):
            renderers.append({"captionTracks": [{**good_track, key: value}]})
        for translation in (None, [None], [{}], [{"languageName": {"runs": []}}]):
            renderers.append(
                {"captionTracks": [good_track], "translationLanguages": translation}
            )
        payloads = [
            b'{"invalid json',
            {},
            [],
            None,
            "bad",
            {"playabilityStatus": {}},
            {"playabilityStatus": None},
            {"playabilityStatus": []},
            {"playabilityStatus": {"status": None}},
            {"playabilityStatus": {"status": 1}},
            {"playabilityStatus": {"status": ""}},
            {"playabilityStatus": {"status": "UNKNOWN"}},
            {"playabilityStatus": {"status": "OK"}, "captions": None},
            {"playabilityStatus": {"status": "OK"}, "captions": []},
        ]
        payloads.extend(
            {
                "playabilityStatus": {"status": "OK"},
                "captions": {"playerCaptionsTracklistRenderer": renderer},
            }
            for renderer in renderers
        )
        for payload in payloads:
            with (
                self.subTest(payload=payload),
                player_payload(payload) as (get, post),
                patch.object(
                    YoutubeDL,
                    "extract_info",
                    return_value={"channel": "Preserved metadata"},
                ) as metadata,
            ):
                now = [0.0]
                client = YouTubeTextClient(clock=lambda now=now: now[0])
                result = client.get("https://youtu.be/abcdefgh_-1")
                self.assertEqual(result["status"], "provider_error")
                self.assertEqual(result["author"], "Preserved metadata")
                self.assertEqual(result["metadata_status"], "ok")
                now[0] = 29
                self.assertEqual(client.get("https://youtu.be/abcdefgh_-1"), result)
                self.assertEqual((get.call_count, post.call_count), (1, 1))
                now[0] = 30
                self.assertEqual(client.get("https://youtu.be/abcdefgh_-1"), result)
                self.assertEqual((get.call_count, post.call_count), (2, 2))
                self.assertEqual(metadata.call_count, 1)

    def test_real_successful_player_without_captions_has_absence_ttl(self):
        for extra in (
            {},
            {"captions": {}},
            {"captions": {"playerCaptionsTracklistRenderer": {}}},
            {"captions": {"playerCaptionsTracklistRenderer": {"captionTracks": []}}},
        ):
            with (
                self.subTest(extra=extra),
                player_payload({"playabilityStatus": {"status": "OK"}, **extra}) as (
                    get,
                    post,
                ),
                patch.object(
                    YoutubeDL, "extract_info", return_value={"title": "Title"}
                ),
            ):
                now = [0.0]
                client = YouTubeTextClient(clock=lambda now=now: now[0])
                result = client.get("https://youtu.be/abcdefgh_-1")
                self.assertEqual(result["status"], "no_transcript")
                self.assertEqual(result["title"], "Title")
                now[0] = 3599
                self.assertEqual(client.get("https://youtu.be/abcdefgh_-1"), result)
                self.assertEqual((get.call_count, post.call_count), (1, 1))
                now[0] = 3600
                client.get("https://youtu.be/abcdefgh_-1")
                self.assertEqual((get.call_count, post.call_count), (2, 2))

    def test_real_player_errors_preserve_classification_and_block_cooldown(self):
        for status, reason, expected in (
            ("ERROR", "This video is unavailable", "unavailable"),
            (
                "LOGIN_REQUIRED",
                "This video may be inappropriate for some users.",
                "unavailable",
            ),
            ("LOGIN_REQUIRED", "Sign in to confirm you’re not a bot", "blocked"),
            ("LOGIN_REQUIRED", "Sign in to confirm you're not a bot", "blocked"),
            ("ERROR", "Sign in to confirm you are not a bot", "blocked"),
            ("ERROR", "Too many requests", "blocked"),
            ("ERROR", "Your IP is likely being blocked by YouTube", "blocked"),
            ("UNPLAYABLE", "Video is private", "unavailable"),
        ):
            with (
                self.subTest(status=status, reason=reason),
                player_payload(
                    {"playabilityStatus": {"status": status, "reason": reason}}
                ) as (get, post),
                patch.object(
                    YoutubeDL, "extract_info", return_value={"title": "Title"}
                ) as metadata,
            ):
                now = [0.0]
                client = YouTubeTextClient(clock=lambda now=now: now[0])
                result = client.get("https://youtu.be/abcdefgh_-1")
                self.assertEqual(result["status"], expected)
                if expected == "blocked":
                    self.assertEqual(result["metadata_status"], "blocked")
                    self.assertEqual(
                        client.get("https://youtu.be/abcdefgh_-2")["status"], "blocked"
                    )
                    self.assertEqual(
                        (get.call_count, post.call_count, metadata.call_count),
                        (1, 1, 0),
                    )
                    now[0] = 299.0
                    client.get("https://youtu.be/abcdefgh_-2")
                    self.assertEqual((get.call_count, post.call_count), (1, 1))
                    now[0] = 300.0
                    client.get("https://youtu.be/abcdefgh_-2")
                    self.assertEqual((get.call_count, post.call_count), (2, 2))
                else:
                    self.assertEqual(result["title"], "Title")

    def test_real_player_http_errors_are_not_schema_failures(self):
        for code, expected in (
            (403, "blocked"),
            (429, "blocked"),
            (503, "transient_error"),
        ):
            with self.subTest(code=code), player_payload(b"not JSON", status_code=code):
                with self.assertRaises(ProviderError) as caught:
                    DefaultYouTubeProvider().transcript("abcdefgh_-1")
                self.assertEqual(caught.exception.status, expected)

    def test_real_listing_and_caption_parser_success(self):
        payload = {
            "playabilityStatus": {"status": "OK"},
            "captions": {
                "playerCaptionsTracklistRenderer": {
                    "captionTracks": [
                        {
                            "baseUrl": "https://www.youtube.com/api/timedtext",
                            "languageCode": "en",
                            "name": {"runs": [{"text": "English"}]},
                            "isTranslatable": True,
                        }
                    ],
                    "translationLanguages": [
                        {
                            "languageCode": "fr",
                            "languageName": {"runs": [{"text": "French"}]},
                        }
                    ],
                }
            },
        }
        with player_payload(
            payload, caption_xml='<transcript><text start="0">Hello</text></transcript>'
        ) as (get, post):
            self.assertEqual(
                DefaultYouTubeProvider().transcript("abcdefgh_-1"), "Hello"
            )
            self.assertEqual((get.call_count, post.call_count), (2, 1))

    def test_cli_real_malformed_listing_emits_json(self):
        from src.youtube_text import main

        output = io.StringIO()
        with (
            player_payload({}),
            patch.object(YoutubeDL, "extract_info", return_value={"title": "Title"}),
            patch("sys.argv", ["youtube_text", "https://youtu.be/abcdefgh_-1"]),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "provider_error")
        self.assertEqual(result["title"], "Title")

    def test_malformed_caption_payload_keeps_metadata_and_negative_cache(self):
        for payload in (
            "<html>temporary upstream error</html",
            '<transcript><text start="invalid" dur="1">Hello</text></transcript>',
            '<transcript><text start="0" dur="invalid">Hello</text></transcript>',
            '<transcript><text dur="1">Hello</text></transcript>',
        ):
            with self.subTest(payload=payload), caption_payload(payload) as request:
                now = [0.0]
                client = YouTubeTextClient(clock=lambda now=now: now[0])
                with patch.object(
                    YoutubeDL,
                    "extract_info",
                    return_value={
                        "channel": "Channel",
                        "description": "Introduction",
                    },
                ) as metadata:
                    result = client.get("https://youtu.be/abcdefgh_-1")
                    self.assertEqual(result["status"], "provider_error")
                    self.assertEqual(result["transcript_status"], "provider_error")
                    self.assertEqual(result["metadata_status"], "ok")
                    self.assertEqual(result["transcript"], "")
                    self.assertEqual(result["description"], "Introduction")
                    self.assertEqual(json.loads(json.dumps(result)), result)
                    self.assertEqual(client.get("https://youtu.be/abcdefgh_-1"), result)
                    self.assertEqual(request.call_count, 1)
                    now[0] = 30
                    client.get("https://youtu.be/abcdefgh_-1")
                    self.assertEqual(request.call_count, 2)
                    self.assertEqual(metadata.call_count, 1)

    def test_cli_malformed_captions_emit_json(self):
        from src.youtube_text import main

        output = io.StringIO()
        with (
            caption_payload("<transcript>"),
            patch.object(YoutubeDL, "extract_info", return_value={"title": "Title"}),
            patch("sys.argv", ["youtube_text", "https://youtu.be/abcdefgh_-1"]),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "provider_error")
        self.assertEqual(result["title"], "Title")
        self.assertEqual(result["metadata_status"], "ok")

    def test_injected_provider_application_errors_propagate(self):
        for error in (ValueError("bug"), KeyError("bug"), RuntimeError("bug")):
            with self.subTest(error=error), self.assertRaises(type(error)):
                YouTubeTextClient(provider=FakeProvider(error)).get(
                    "https://youtu.be/abcdefgh_-1"
                )

    def test_real_metadata_ip_block_cools_down_other_videos_keeps_cached_data(self):
        from yt_dlp.utils import DownloadError

        client = YouTubeTextClient(clock=lambda: 0.0)
        with (
            caption_payload(
                '<transcript><text start="0">Hello</text></transcript>'
            ) as captions,
            patch.object(
                YoutubeDL,
                "extract_info",
                side_effect=[
                    {"title": "Cached title"},
                    DownloadError(
                        "All player responses are invalid. "
                        "Your IP is likely being blocked by Youtube"
                    ),
                ],
            ) as metadata,
        ):
            cached = client.get("https://youtu.be/abcdefgh_-1")
            self.assertEqual(cached["status"], "ok")
            blocked = client.get("https://youtu.be/abcdefgh_-2")
            self.assertEqual(blocked["metadata_status"], "blocked")
            self.assertEqual(blocked["transcript"], "Hello")
            self.assertEqual(blocked["status"], "partial")
            self.assertEqual(
                client.get("https://youtu.be/abcdefgh_-3")["status"], "blocked"
            )
            self.assertEqual(client.get("https://youtu.be/abcdefgh_-1"), cached)
            self.assertEqual(client.get("https://youtu.be/abcdefgh_-2"), blocked)
            self.assertEqual(captions.call_count, 2)
            self.assertEqual(metadata.call_count, 2)

    def test_captions_choose_available_language_without_second_listing(self):
        track = SimpleNamespace(
            language_code="fr",
            is_generated=False,
            fetch=lambda: [
                SimpleNamespace(text="Bonjour"),
                SimpleNamespace(text="monde"),
            ],
        )
        api = MagicMock()
        api.list.return_value = [track]
        with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=api):
            self.assertEqual(
                DefaultYouTubeProvider().transcript("abcdefgh_-1"), "Bonjour monde"
            )
        api.list.assert_called_once_with("abcdefgh_-1")

    def test_regional_english_precedes_other_languages(self):
        tracks = [
            SimpleNamespace(
                language_code=code,
                is_generated=generated,
                fetch=lambda text=text: [SimpleNamespace(text=text)],
            )
            for code, generated, text in [
                ("fr", False, "French"),
                ("en", True, "Generated English"),
                ("en-US", False, "Manual English"),
            ]
        ]
        api = MagicMock()
        api.list.return_value = tracks
        with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=api):
            self.assertEqual(
                DefaultYouTubeProvider().transcript("abcdefgh_-1"), "Manual English"
            )

    def test_typed_caption_failures(self):
        from youtube_transcript_api import (
            IpBlocked,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        for error, status in (
            (IpBlocked("abcdefgh_-1"), "blocked"),
            (TranscriptsDisabled("abcdefgh_-1"), "no_transcript"),
            (VideoUnavailable("abcdefgh_-1"), "unavailable"),
        ):
            api = MagicMock()
            api.list.side_effect = error
            with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=api):
                with self.assertRaises(ProviderError) as caught:
                    DefaultYouTubeProvider().transcript("abcdefgh_-1")
                self.assertEqual(caught.exception.status, status)

    def test_no_tracks_and_empty_response_differ(self):
        api = MagicMock()
        api.list.return_value = []
        with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=api):
            with self.assertRaises(ProviderError) as caught:
                DefaultYouTubeProvider().transcript("abcdefgh_-1")
            self.assertEqual(caught.exception.status, "no_transcript")

    def test_wrapped_http_errors_preserve_status(self):
        import requests
        from youtube_transcript_api import YouTubeRequestFailed

        for code, status in (
            (403, "blocked"),
            (503, "transient_error"),
            (400, "provider_error"),
        ):
            response = requests.Response()
            response.status_code = code
            api = MagicMock()
            try:
                raise requests.HTTPError(f"{code} server error", response=response)
            except requests.HTTPError as cause:
                try:
                    raise YouTubeRequestFailed("abcdefgh_-1", cause)
                except YouTubeRequestFailed as wrapped:
                    api.list.side_effect = wrapped
            with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=api):
                with self.assertRaises(ProviderError) as caught:
                    DefaultYouTubeProvider().transcript("abcdefgh_-1")
                self.assertEqual(caught.exception.status, status)

    def test_transport_timeout_is_transient(self):
        import requests

        api = MagicMock()
        api.list.side_effect = requests.Timeout("network")
        with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=api):
            with self.assertRaises(ProviderError) as caught:
                DefaultYouTubeProvider().transcript("abcdefgh_-1")
            self.assertEqual(caught.exception.status, "transient_error")

    def test_metadata_mapping_canonical_url_and_no_download_options(self):
        with patch("yt_dlp.YoutubeDL") as factory:
            extractor = factory.return_value.__enter__.return_value
            extractor.extract_info.return_value = {
                "channel": "Channel",
                "uploader": "Uploader",
                "description": "Introduction",
                "title": "Title",
                "tags": ["health"],
            }
            result = DefaultYouTubeProvider().metadata("abcdefgh_-1")
            self.assertEqual(
                result,
                {
                    "author": "Channel",
                    "description": "Introduction",
                    "title": "Title",
                    "keywords": ["health"],
                },
            )
            extractor.extract_info.assert_called_once_with(
                "https://www.youtube.com/watch?v=abcdefgh_-1", download=False
            )
            options = factory.call_args.args[0]
            self.assertTrue(options["skip_download"])
            self.assertEqual(options["retries"], 0)
            self.assertEqual(options["extractor_retries"], 0)
            self.assertEqual(options["socket_timeout"], 10)

    def test_metadata_block_is_not_no_transcript(self):
        from yt_dlp.utils import DownloadError

        for message in (
            "HTTP Error 429: Too Many Requests",
            "HTTP Error 403: Forbidden",
            "Sign in to confirm you're not a bot",
            "Sign in to confirm you’re not a bot",
        ):
            with self.subTest(message=message), patch("yt_dlp.YoutubeDL") as factory:
                factory.return_value.__enter__.return_value.extract_info.side_effect = (
                    DownloadError(message)
                )
                with self.assertRaises(ProviderError) as caught:
                    DefaultYouTubeProvider().metadata("abcdefgh_-1")
                self.assertEqual(caught.exception.status, "blocked")

    def test_caption_http_timeout_is_bounded_and_proxy_environment_disabled(self):
        import requests

        def make_api(*, http_client):
            self.assertFalse(http_client.trust_env)
            with patch.object(
                requests.Session, "request", return_value=object()
            ) as request:
                http_client.request("GET", "https://www.youtube.com/")
                self.assertEqual(request.call_args.kwargs["timeout"], 10)
            return SimpleNamespace(list=lambda video_id: [])

        with (
            patch("youtube_transcript_api.YouTubeTranscriptApi", side_effect=make_api),
            self.assertRaises(ProviderError),
        ):
            DefaultYouTubeProvider().transcript("abcdefgh_-1")

    def test_missing_optional_packages_have_explicit_status(self):
        import builtins

        original_import = builtins.__import__

        def without_optional(name, *args, **kwargs):
            if name in ("youtube_transcript_api", "yt_dlp"):
                raise ImportError("Not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_optional):
            result = YouTubeTextClient().get("https://youtu.be/abcdefgh_-1")
        self.assertEqual(result["status"], "dependency_missing")
        self.assertEqual(result["metadata_status"], "dependency_missing")

    def test_metadata_other_failures_and_missing_fields(self):
        from yt_dlp.utils import DownloadError

        for message, status in (
            ("Private video", "unavailable"),
            ("Connection timed out", "transient_error"),
            ("Extractor failed", "provider_error"),
        ):
            with patch("yt_dlp.YoutubeDL") as factory:
                factory.return_value.__enter__.return_value.extract_info.side_effect = (
                    DownloadError(message)
                )
                with self.assertRaises(ProviderError) as caught:
                    DefaultYouTubeProvider().metadata("abcdefgh_-1")
                self.assertEqual(caught.exception.status, status)
        with patch("yt_dlp.YoutubeDL") as factory:
            factory.return_value.__enter__.return_value.extract_info.return_value = {
                "uploader": "Uploader"
            }
            self.assertEqual(
                DefaultYouTubeProvider().metadata("abcdefgh_-1"),
                {"author": "Uploader", "title": "", "description": "", "keywords": []},
            )

    def test_adapters_reject_raw_ids_before_network(self):
        for method in ("metadata", "transcript"):
            with self.assertRaises(ValueError):
                getattr(DefaultYouTubeProvider(), method)("https://evil.test/")


class FakeProvider:
    def __init__(
        self,
        captions: str | Exception = "Hello world",
        metadata: dict[str, Any] | Exception | None = None,
    ):
        self.captions = captions
        self.info = (
            metadata
            if metadata is not None
            else {
                "author": "Channel",
                "title": "Title",
                "description": "Introduction",
                "keywords": ["health", "science"],
            }
        )
        self.calls: list[tuple[str, str]] = []

    def transcript(self, video_id):
        self.calls.append(("transcript", video_id))
        if isinstance(self.captions, Exception):
            raise self.captions
        return self.captions

    def metadata(self, video_id):
        self.calls.append(("metadata", video_id))
        if isinstance(self.info, Exception):
            raise self.info
        return self.info


class ClientTests(unittest.TestCase):
    url = "https://youtu.be/abcdefgh_-1"

    def test_cache_ttl_starts_after_slow_provider_outcome(self):
        for outcome, ttl in (
            ("Hello", 3600),
            (ProviderError("no_transcript"), 3600),
            (ProviderError("unavailable"), 3600),
            (ProviderError("provider_error"), 30),
            (ProviderError("transient_error"), 30),
        ):
            with self.subTest(outcome=outcome):
                now = [0.0]

                class SlowProvider(FakeProvider):
                    def transcript(self, video_id, now=now):
                        now[0] += 4000
                        return super().transcript(video_id)

                provider = SlowProvider(outcome)
                client = YouTubeTextClient(
                    provider=provider, clock=lambda now=now: now[0]
                )
                result = client.get(self.url)
                completed_at = now[0]
                now[0] = completed_at + ttl - 1
                self.assertEqual(client.get(self.url), result)
                self.assertEqual(len(provider.calls), 2)
                now[0] = completed_at + ttl
                client.get(self.url)
                self.assertEqual(
                    [stage for stage, _ in provider.calls].count("transcript"), 2
                )

    def test_slow_block_starts_full_cooldown_and_immediately_skips_metadata(self):
        now = [0.0]

        class SlowProvider(FakeProvider):
            def transcript(self, video_id):
                now[0] += 301
                return super().transcript(video_id)

        provider = SlowProvider(ProviderError("blocked"))
        client = YouTubeTextClient(provider=provider, clock=lambda now=now: now[0])
        result = client.get(self.url)
        self.assertEqual(result["metadata_status"], "blocked")
        self.assertEqual(provider.calls, [("transcript", "abcdefgh_-1")])
        now[0] = 600
        self.assertEqual(client.get(self.url)["status"], "blocked")
        self.assertEqual(
            client.get("https://youtu.be/abcdefgh_-2")["status"], "blocked"
        )
        self.assertEqual(len(provider.calls), 1)
        now[0] = 601
        provider.captions = "Recovered"
        self.assertEqual(client.get(self.url)["status"], "ok")
        self.assertEqual(len(provider.calls), 3)

    def test_cli_invalid_url_emits_json_and_nonzero_exit(self):
        import json
        import subprocess
        import sys

        completed = subprocess.run(
            [sys.executable, "-m", "src.youtube_text", "https://evil.test/"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(json.loads(completed.stdout)["status"], "invalid_url")
        self.assertEqual(completed.stderr, "")

    def test_invalid_input_never_calls_provider(self):
        provider = FakeProvider()
        result = YouTubeTextClient(provider=provider).get("https://evil.test/")
        self.assertEqual(result["status"], "invalid_url")
        self.assertEqual(result["video_id"], "")
        self.assertEqual(result["transcript"], "")
        self.assertEqual(provider.calls, [])

    def test_success_metadata_and_same_video_cache_are_copy_safe(self):
        provider = FakeProvider()
        client = YouTubeTextClient(provider=provider)
        result = client.get(self.url)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["author"], "Channel")
        self.assertEqual(result["transcript"], "Hello world")
        self.assertEqual(result["description"], "Introduction")
        self.assertEqual(result["title"], "Title")
        self.assertEqual(result["keywords"], ["health", "science"])
        result["keywords"].append("mutated")
        self.assertEqual(
            client.get("https://youtube.com/watch?v=abcdefgh_-1")["keywords"],
            ["health", "science"],
        )
        self.assertEqual(len(provider.calls), 2)

    def test_caption_failures_keep_metadata_and_distinguish_semantics(self):
        for status in (
            "no_transcript",
            "unavailable",
            "transient_error",
            "provider_error",
            "dependency_missing",
        ):
            with self.subTest(status=status):
                result = YouTubeTextClient(
                    provider=FakeProvider(ProviderError(status))
                ).get(self.url)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["transcript_status"], status)
                self.assertEqual(result["description"], "Introduction")
                self.assertEqual(result["transcript"], "")

    def test_metadata_failure_preserves_transcript(self):
        result = YouTubeTextClient(
            provider=FakeProvider(metadata=ProviderError("blocked"))
        ).get(self.url)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["metadata_status"], "blocked")
        self.assertEqual(result["transcript"], "Hello world")
        self.assertEqual(result["author"], "")
        self.assertEqual(result["keywords"], [])

    def test_blocked_caption_skips_metadata_and_cools_down_all_videos(self):
        now = [0.0]
        provider = FakeProvider(ProviderError("blocked"))
        client = YouTubeTextClient(provider=provider, clock=lambda now=now: now[0])
        self.assertEqual(client.get(self.url)["status"], "blocked")
        self.assertEqual(
            client.get("https://youtu.be/abcdefgh_-2")["status"], "blocked"
        )
        self.assertEqual(len(provider.calls), 1)
        now[0] = 301
        provider.captions = "Recovered"
        self.assertEqual(client.get(self.url)["status"], "ok")

    def test_transient_negative_cache_then_retry_reuses_metadata(self):
        now = [0.0]
        provider = FakeProvider(ProviderError("transient_error"))
        client = YouTubeTextClient(provider=provider, clock=lambda now=now: now[0])
        client.get(self.url)
        client.get(self.url)
        self.assertEqual(len(provider.calls), 2)
        now[0] = 31
        provider.captions = "Recovered"
        self.assertEqual(client.get(self.url)["status"], "ok")
        self.assertEqual(
            [x[0] for x in provider.calls], ["transcript", "metadata", "transcript"]
        )

    def test_empty_captions_are_not_evidence_of_absence(self):
        self.assertEqual(
            YouTubeTextClient(provider=FakeProvider("")).get(self.url)["status"],
            "provider_error",
        )

    def test_missing_metadata_is_not_fabricated(self):
        result = YouTubeTextClient(provider=FakeProvider(metadata={})).get(self.url)
        self.assertEqual(result["author"], "")
        self.assertEqual(result["description"], "")
        self.assertEqual(result["keywords"], [])

    def test_cache_expiry_and_capacity(self):
        now = [0.0]
        provider = FakeProvider()
        client = YouTubeTextClient(
            provider=provider, clock=lambda now=now: now[0], cache_size=1
        )
        client.get(self.url)
        client.get("https://youtu.be/abcdefgh_-2")
        client.get(self.url)
        self.assertEqual(len(provider.calls), 6)
        now[0] = 3601
        client.get(self.url)
        self.assertEqual(len(provider.calls), 8)


class URLTests(unittest.TestCase):
    def test_duplicate_v_rejected_for_every_path_family_without_provider_calls(self):
        for base in (
            "https://youtube.com/watch",
            "https://youtu.be/abcdefgh_-1",
            "https://youtube.com/shorts/abcdefgh_-1",
            "https://youtube.com/embed/abcdefgh_-1",
            "https://youtube.com/live/abcdefgh_-1",
        ):
            with self.subTest(base=base):
                provider = FakeProvider()
                url = base + "?v=abcdefgh_-1&v=abcdefgh_-2"
                self.assertIsNone(normalize_url(url))
                result = YouTubeTextClient(provider=provider).get(url)
                self.assertEqual(result["status"], "invalid_url")
                self.assertEqual(provider.calls, [])

    def test_normalizes_supported_urls(self):
        for url in (
            "https://www.youtube.com/watch?v=abcdefgh_-1&t=9",
            "http://m.youtube.com/watch?v=abcdefgh_-1",
            "https://youtu.be/abcdefgh_-1?si=share",
            "https://youtube.com/shorts/abcdefgh_-1",
            "https://youtube.com/embed/abcdefgh_-1",
            "https://youtube.com/live/abcdefgh_-1",
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    normalize_url(url),
                    ("abcdefgh_-1", "https://www.youtube.com/watch?v=abcdefgh_-1"),
                )

    def test_rejects_invalid_or_ambiguous_urls(self):
        for url in (
            "abcdefgh_-1",
            "",
            None,
            "https://evil.test/watch?v=abcdefgh_-1",
            "https://youtube.com.evil.test/watch?v=abcdefgh_-1",
            "https://evil.test/youtu.be/abcdefgh_-1",
            "https://user@youtube.com/watch?v=abcdefgh_-1",
            "https://youtube.com:443/watch?v=abcdefgh_-1",
            "ftp://youtube.com/watch?v=abcdefgh_-1",
            "https://youtube.com/watch?v=abcdefgh_-12",
            "https://youtube.com/watch?v=abcdefgh_-1&v=abcdefgh_-2",
            "https://youtu.be/abcdefgh_-1/extra",
            "https://youtube.com/watch?v=short",
            "https://youtube.com/watch?v=abcdefgh_%311",
            "https://youtube.com/playlist?list=abcdefgh_-1",
        ):
            with self.subTest(url=url):
                self.assertIsNone(normalize_url(url))


if __name__ == "__main__":
    unittest.main()
