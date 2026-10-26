"""
Tests for extract_local_files() — auto-detection of bare local file paths
in model response text for native media delivery.

Covers: path matching, code-block exclusion, URL rejection, tilde expansion,
deduplication, text cleanup, and extension routing.

Based on PR #1636 by sudoingX (salvaged + hardened).
"""

from unittest.mock import patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(content: str, existing_files: set[str] | None = None):
    """
    Run extract_local_files with os.path.isfile mocked to return True
    for any path in *existing_files* (expanded form).  If *existing_files*
    is None every path passes.
    """
    existing = existing_files

    def fake_isfile(p):
        if existing is None:
            return True
        return p in existing

    def fake_expanduser(p):
        if p.startswith("~/"):
            return "/home/user" + p[1:]
        return p

    with patch("os.path.isfile", side_effect=fake_isfile), \
         patch("os.path.expanduser", side_effect=fake_expanduser):
        return BasePlatformAdapter.extract_local_files(content)


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

class TestBasicDetection:

    def test_absolute_path_image(self):
        paths, cleaned = _extract("Here is the screenshot /root/screenshots/game.png enjoy")
        assert paths == ["/root/screenshots/game.png"]
        assert "/root/screenshots/game.png" not in cleaned
        assert "Here is the screenshot" in cleaned

    def test_tilde_path_image(self):
        paths, cleaned = _extract("Check out ~/photos/cat.jpg for the cat")
        assert paths == ["/home/user/photos/cat.jpg"]
        assert "~/photos/cat.jpg" not in cleaned

    def test_video_extensions(self):
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            text = f"Video at /tmp/clip{ext} here"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/clip{ext}"

    def test_image_extensions(self):
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            text = f"Image at /tmp/pic{ext} here"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/pic{ext}"

    def test_document_extensions(self):
        """Documents (PDF, Word, plain text, etc.) ship as file uploads."""
        for ext in (".pdf", ".docx", ".doc", ".odt", ".rtf", ".txt", ".md"):
            text = f"Report at /tmp/report{ext} attached"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/report{ext}"


    def test_path_at_line_start(self):
        paths, _ = _extract("/var/data/image.png")
        assert paths == ["/var/data/image.png"]


# ---------------------------------------------------------------------------
# Non-existent files are skipped
# ---------------------------------------------------------------------------

class TestIsfileGuard:

    def test_nonexistent_path_skipped(self):
        """Paths that don't exist on disk are not extracted."""
        paths, cleaned = _extract(
            "See /tmp/nope.png here",
            existing_files=set(),  # nothing exists
        )
        assert paths == []
        assert "/tmp/nope.png" in cleaned  # not stripped


# ---------------------------------------------------------------------------
# URL false-positive prevention
# ---------------------------------------------------------------------------

class TestURLRejection:

    def test_https_url_not_matched(self):
        """Paths embedded in HTTP URLs must not be extracted."""
        paths, cleaned = _extract("Visit https://example.com/images/photo.png for details")
        # The regex lookbehind should prevent matching the URL's path segment
        # Even if it did match, isfile would be False for /images/photo.png
        # (we mock isfile to True-for-all here, so the lookbehind is the guard)
        assert paths == []
        assert "https://example.com/images/photo.png" in cleaned


# ---------------------------------------------------------------------------
# Code block exclusion
# ---------------------------------------------------------------------------

class TestCodeBlockExclusion:

    def test_fenced_code_block_skipped(self):
        text = "Here's how:\n```python\nimg = open('/tmp/image.png')\n```\nDone."
        paths, cleaned = _extract(text)
        assert paths == []
        assert "/tmp/image.png" in cleaned  # not stripped

    def test_inline_code_skipped(self):
        text = "Use the path `/tmp/image.png` in your config"
        paths, cleaned = _extract(text)
        assert paths == []
        assert "`/tmp/image.png`" in cleaned


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_duplicate_paths_deduplicated(self):
        text = "See /tmp/img.png and also /tmp/img.png again"
        paths, _ = _extract(text)
        assert paths == ["/tmp/img.png"]


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

class TestTextCleanup:

    def test_path_removed_from_text(self):
        _paths, cleaned = _extract("Before /tmp/x.png after")
        assert "Before" in cleaned
        assert "after" in cleaned
        assert "/tmp/x.png" not in cleaned

    def test_excessive_blank_lines_collapsed(self):
        text = "Before\n\n\n/tmp/x.png\n\n\nAfter"
        _, cleaned = _extract(text)
        assert "\n\n\n" not in cleaned


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string(self):
        paths, cleaned = _extract("")
        assert paths == []
        assert cleaned == ""


    @pytest.mark.parametrize(
        "content,expected",
        [
            # Backslash separators (native Windows style)
            ("See C:\\Users\\test\\image.png here", "C:\\Users\\test\\image.png"),
            # Forward slashes with drive letter (common in cross-platform code)
            ("See C:/Users/test/image.png here", "C:/Users/test/image.png"),
            # Non-C: drive
            ("Video at D:/data/clip.mp4 ready", "D:/data/clip.mp4"),
            # Lowercase drive letter
            ("Path e:/audio/track.mp3 done", "e:/audio/track.mp3"),
        ],
    )
    def test_windows_drive_letter_paths_matched(self, content, expected):
        """Windows drive-letter paths (C:/..., C:\\...) must be detected (#34632).

        Prior behavior anchored on (?:~/|/) only, which silently dropped
        Windows absolute paths so the agent's bare-path references were
        sent as text instead of native uploads.
        """
        paths, cleaned = _extract(content)
        assert paths == [expected]
        assert expected not in cleaned

    def test_relative_windows_path_not_matched(self):
        """A bare Windows-style filename without a drive letter must still
        not match (e.g. ``foo\\bar.png`` is treated as relative, like its
        Unix sibling ``foo/bar.png``)."""
        paths, _ = _extract("File at foo\\bar.png here")
        assert paths == []


    def test_path_followed_by_punctuation(self):
        """Path followed by comma, period, paren should still match."""
        for suffix in [",", ".", ")", ":", ";"]:
            text = f"See /tmp/img.png{suffix} details"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed with suffix '{suffix}'"


# ---------------------------------------------------------------------------
# Reply text: only a path that is attached leaves it
# ---------------------------------------------------------------------------

class _ReplyAdapter(BasePlatformAdapter):

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.DISCORD)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


async def _extract_reply(adapter, reply: str):
    source = SessionSource(platform=Platform.DISCORD, chat_id="111", chat_type="dm")
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="m1")
    return await adapter._extract_response_content(reply, event, "session-key", is_ephemeral_response=False)


class TestUnattachedPathStaysInReply:

    @pytest.mark.asyncio
    async def test_refused_path_stays_while_attached_path_is_cut(self, tmp_path, monkeypatch):
        # ~/.config is on the delivery denylist, so the settings file is never attached: cutting it
        # from the sentence left the user with neither the file nor where it lives.
        monkeypatch.setenv("HOME", str(tmp_path))
        settings = tmp_path / ".config" / "app" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text("{}")
        report = tmp_path / "report.pdf"
        report.write_bytes(b"%PDF-1.4")
        reply = f"I set dark mode in ~/.config/app/settings.json. Summary: {report}."
        out = await _extract_reply(_ReplyAdapter(), reply)
        assert out.local_files == [str(report.resolve())]
        assert out.text_content == "I set dark mode in ~/.config/app/settings.json. Summary: ."

    @pytest.mark.asyncio
    async def test_path_already_delivered_this_session_stays(self, tmp_path, monkeypatch):
        chart = tmp_path / "chart.png"
        chart.write_bytes(b"\x89PNG")
        adapter = _ReplyAdapter()

        async def _history(_session_key):
            return {str(chart.resolve())}

        monkeypatch.setattr(adapter, "_bounded_history_media_paths_for_session", _history)
        out = await _extract_reply(adapter, f"Sales rose, as {chart} shows.")
        assert out.local_files == []
        assert out.text_content == f"Sales rose, as {chart} shows."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
