"""Offline regressions for intermediate block signals and cooldown cache pressure."""

import unittest
from unittest.mock import Mock, patch

from src.youtube_text import DefaultYouTubeProvider, ProviderError, YouTubeTextClient
from tests.test_youtube_text import FakeProvider


class BlockSignalTests(unittest.TestCase):
    def test_intermediate_block_warning_aborts_extraction_and_other_videos(self):
        from yt_dlp import YoutubeDL

        after_warning = Mock()

        def extract(extractor, url, download=False):
            extractor.report_warning("HTTP Error 429: Too Many Requests")
            after_warning()
            return {"channel": "Should not be reached"}

        with (
            patch.object(
                DefaultYouTubeProvider, "transcript", return_value="Captions"
            ) as captions,
            patch.object(YoutubeDL, "extract_info", extract),
        ):
            client = YouTubeTextClient(clock=lambda: 0.0)
            first = client.get("https://youtu.be/abcdefgh_-1")
            self.assertEqual(first["transcript"], "Captions")
            self.assertEqual(first["metadata_status"], "blocked")
            self.assertEqual(
                client.get("https://youtu.be/abcdefgh_-2")["status"], "blocked"
            )
            captions.assert_called_once()
            after_warning.assert_not_called()

    def test_video_id_digits_are_not_http_block_statuses(self):
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError

        for message, expected in (
            ("[youtube] abc403defgh: Private video", "unavailable"),
            ("[youtube] abc429defgh: Connection timed out", "transient_error"),
        ):
            with (
                self.subTest(message=message),
                patch.object(
                    YoutubeDL, "extract_info", side_effect=DownloadError(message)
                ),
            ):
                with self.assertRaises(ProviderError) as caught:
                    DefaultYouTubeProvider().metadata("abcdefgh_-1")
                self.assertEqual(caught.exception.status, expected)

    def test_non_block_warning_allows_metadata(self):
        from yt_dlp import YoutubeDL

        def extract(extractor, url, download=False):
            extractor.report_warning("Optional JavaScript runtime is not available")
            return {"channel": "Channel"}

        with patch.object(YoutubeDL, "extract_info", extract):
            self.assertEqual(
                DefaultYouTubeProvider().metadata("abcdefgh_-1")["author"], "Channel"
            )

    def test_suppressed_uncached_video_does_not_evict_success(self):
        provider = FakeProvider()
        client = YouTubeTextClient(provider, clock=lambda: 0.0, cache_size=2)
        first = client.get("https://youtu.be/abcdefgh_-1")
        provider.captions = ProviderError("blocked")
        client.get("https://youtu.be/abcdefgh_-2")
        before = len(provider.calls)
        client.get("https://youtu.be/abcdefgh_-3")
        self.assertEqual(client.get("https://youtu.be/abcdefgh_-1"), first)
        self.assertEqual(len(provider.calls), before)
