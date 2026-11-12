# Copyright: Ankitects Pty Ltd and contributors
# License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

import shutil
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import aqt
from anki.sound import SoundOrVideoTag
from anki.utils import is_lin, is_mac, is_win
from aqt.mpv import MPV
from aqt.sound import MpvManager, _packagedCmd, is_audio_file


def test_is_audio_file_recognizes_common_formats():
    for ext in ("mp3", "wav", "ogg", "flac", "m4a", "opus", "spx", "oga"):
        assert is_audio_file(f"test.{ext}")


def test_is_audio_file_is_case_insensitive():
    for ext in ("MP3", "WAV", "OGG", "FLAC"):
        assert is_audio_file(f"test.{ext}")


def test_is_audio_file_rejects_non_audio():
    # mp4/avi are video-only; jpg/png/pdf are not media Anki plays via mpv.
    for ext in ("mp4", "avi", "jpg", "png", "pdf"):
        assert not is_audio_file(f"test.{ext}")


def test_is_audio_file_rejects_no_extension():
    assert not is_audio_file("audiofile")


def test_packagedcmd_returns_absolute_path_when_anki_audio_available():
    # _packagedCmd should prefer the binary bundled in anki_audio over a
    # system-wide one on macOS and Windows. This is the regression caught by
    # issue #5015: an updated anki_audio build was not being picked up.
    if not (is_mac or is_win):
        pytest.skip("anki_audio binary preference is only used on macOS/Windows")

    try:
        from pathlib import Path as _Path

        import anki_audio

        audio_pkg_path = _Path(anki_audio.__file__).parent
    except ImportError:
        pytest.skip("anki_audio package not installed")

    cmd, _env = _packagedCmd(["mpv"])
    mpv_path = Path(cmd[0])

    assert mpv_path.is_absolute(), "expected an absolute path to the packaged binary"
    assert str(audio_pkg_path) in str(mpv_path), (
        f"expected binary inside anki_audio package at {audio_pkg_path}, got {mpv_path}"
    )


def _resolved_mpv_command(args: list[str]) -> tuple[list[str], dict[str, str]]:
    cmd, env = _packagedCmd(args)
    mpv = cmd[0]
    if not Path(mpv).is_absolute():
        mpv = shutil.which(mpv)
        if mpv is None:
            pytest.skip("mpv not found")
        cmd[0] = mpv

    return cmd, env


def test_mpv_binary_runs():
    cmd, env = _resolved_mpv_command(["mpv"])
    result = subprocess.run(cmd + ["--version"], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


@pytest.fixture
def generated_wav(tmp_path: Path) -> Path:
    wav_path = tmp_path / "silence.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44_100)
        wav.writeframes(b"\0\0" * 4_410)
    return wav_path


def test_mpv_can_play_generated_wav(generated_wav: Path):
    cmd, env = _resolved_mpv_command(
        [
            "mpv",
            "--no-terminal",
            "--force-window=no",
            "--audio-display=no",
            "--keep-open=no",
            "--autoload-files=no",
            "--ao=null",
            "--vo=null",
            "--",
            str(generated_wav),
        ]
    )

    result = subprocess.run(cmd, env=env, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.skipif(is_lin, reason="mpv is not bundled for Linux")
def test_mpvmanager_can_play_generated_wav(
    monkeypatch, tmp_path: Path, generated_wav: Path
):
    monkeypatch.setattr(
        MpvManager, "default_argv", MpvManager.default_argv + ["--ao=null", "--vo=null"]
    )
    mock_mw = MagicMock()
    mock_mw.taskman.run_in_background.side_effect = (
        lambda task, on_done=None, **kwargs: task()
    )
    monkeypatch.setattr(aqt, "mw", mock_mw)
    manager = MpvManager(tmp_path, tmp_path)
    manager.play(SoundOrVideoTag(filename=str(generated_wav.name)), lambda _: None)


def _mpv_with_callbacks(callbacks: dict) -> MPV:
    # Skip __init__ so no mpv process or background registration is started.
    player = MPV.__new__(MPV)
    player._callbacks_initialized = True
    player._callbacks = callbacks
    return player


def _failing_callback(*args) -> None:
    raise RuntimeError("callback failed")


@pytest.mark.parametrize(
    "callback_name, message, expected",
    [
        ("file-loaded", {"event": "file-loaded"}, ("handled",)),
        (
            "property-idle-active",
            {"event": "property-change", "name": "idle-active", "data": True},
            ("handled", True),
        ),
    ],
)
def test_mpv_event_runs_remaining_callbacks_when_one_raises(
    capsys, callback_name: str, message: dict, expected: tuple
):
    calls = []
    player = _mpv_with_callbacks(
        {
            callback_name: [
                _failing_callback,
                lambda *args: calls.append(("handled", *args)),
            ]
        }
    )

    player._handle_event(message)

    assert calls == [expected]
    assert "callback failed" in capsys.readouterr().err


class _StopEventReader(BaseException):
    """Ends the otherwise endless event loop; not an Exception, so the
    per-callback guard cannot swallow it."""


def test_mpv_event_thread_keeps_dispatching_after_a_callback_raises(capsys):
    calls = []

    def fail_on_first_load() -> None:
        if not calls:
            calls.append("failed")
            raise RuntimeError("callback failed")
        calls.append("loaded")

    player = _mpv_with_callbacks({"file-loaded": [fail_on_first_load]})
    events = iter([{"event": "file-loaded"}, None, {"event": "file-loaded"}])

    def next_event(timeout=None):
        try:
            return next(events)
        except StopIteration:
            raise _StopEventReader

    player._get_event = next_event

    with pytest.raises(_StopEventReader):
        player._event_reader()

    assert calls == ["failed", "loaded"]
    assert "callback failed" in capsys.readouterr().err


def _mpvmanager_with_completion(on_done) -> MpvManager:
    # Skip __init__ so no mpv process is started.
    manager = MpvManager.__new__(MpvManager)
    manager._on_done = on_done
    return manager


def test_mpvmanager_idle_does_not_raise_when_main_window_cleared(monkeypatch):
    manager = _mpvmanager_with_completion(MagicMock())
    monkeypatch.setattr(aqt, "mw", None)

    # mpv can report idle after shutdown has cleared the main window; this
    # must not raise into the event thread.
    manager.on_property_idle_active(True)


def test_mpvmanager_idle_runs_completion_on_main_thread(monkeypatch):
    on_done = MagicMock()
    manager = _mpvmanager_with_completion(on_done)
    mock_mw = MagicMock()
    monkeypatch.setattr(aqt, "mw", mock_mw)

    manager.on_property_idle_active(True)

    mock_mw.taskman.run_on_main.assert_called_once_with(on_done)
    on_done.assert_not_called()


@pytest.mark.parametrize(
    "idle, on_done",
    [(False, MagicMock()), (True, None)],
    ids=["playing", "no_completion"],
)
def test_mpvmanager_idle_schedules_nothing_when_playing_or_no_completion(
    monkeypatch, idle: bool, on_done
):
    manager = _mpvmanager_with_completion(on_done)
    mock_mw = MagicMock()
    monkeypatch.setattr(aqt, "mw", mock_mw)

    manager.on_property_idle_active(idle)

    mock_mw.taskman.run_on_main.assert_not_called()
