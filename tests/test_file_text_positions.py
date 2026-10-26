import io
import pickle

import pytest

import cloudpickle


@pytest.mark.parametrize(
    "encoding, text, count, newline",
    [
        ("utf-8", "\u00e9abc", 1, None),
        ("utf-8", "a\u2603bc", 2, None),
        ("utf-16", "abcd", 1, None),
        ("utf-32", "abcd", 1, None),
        ("utf-8-sig", "abcd", 1, None),
        ("utf-8", "first\r\nlast", 6, None),
        ("utf-8", "first\rlast", 6, None),
        ("utf-8", "first\r\nlast", 7, ""),
        ("utf-8", "first\r\nlast", 6, ""),
        ("utf-8", "\u00e9\r\nlast", 3, ""),
        ("utf-16", "first\r\nlast", 7, ""),
        ("latin-1", "\u00e9abc", 1, None),
        ("ascii", "abcd", 1, None),
        ("utf-16", "abcd", 0, None),
        ("utf-8", "", 0, None),
        ("utf-8", "\u00e9abc", 4, None),
        ("utf-16", "abcd", 4, None),
        ("utf-8", "first\r", 6, None),
    ],
)
def test_text_file_roundtrip_preserves_character_position(
    tmp_path, encoding, text, count, newline
):
    path = tmp_path / "text.txt"
    path.write_bytes(text.encode(encoding))
    with path.open(encoding=encoding, newline=newline) as source:
        prefix = source.read(count)
        cookie = source.tell()
        restored = pickle.loads(cloudpickle.dumps(source))
        assert source.tell() == cookie
        assert restored.tell() == len(prefix)
        assert restored.read() == source.read()
        source.seek(0)
        restored.seek(0)
        assert restored.read() == source.read()


@pytest.mark.parametrize("encoding", ["ascii", "utf-8", "utf-16", "utf-32"])
def test_text_file_roundtrip_preserves_existing_position_beyond_eof(tmp_path, encoding):
    path = tmp_path / "text.txt"
    path.write_bytes("abcd".encode(encoding))
    with path.open(encoding=encoding) as source:
        source.seek(100)
        restored = pickle.loads(cloudpickle.dumps(source))
        assert source.tell() == 100
        assert restored.tell() == 100
        assert restored.read() == source.read() == ""


def test_stringio_roundtrip_preserves_character_position():
    source = io.StringIO("\u00e9abc")
    source.read(1)
    restored = pickle.loads(cloudpickle.dumps(source))
    assert source.tell() == restored.tell() == 1
    assert restored.read() == source.read() == "abc"


def test_text_file_position_is_restored_on_decode_error(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"prefix\xff")
    with path.open(encoding="utf-8") as source:
        source.seek(2)
        cookie = source.tell()
        with pytest.raises(UnicodeDecodeError):
            cloudpickle.dumps(source)
        assert source.tell() == cookie


def test_text_file_position_is_restored_when_full_read_fails(tmp_path):
    path = tmp_path / "invalid-prefix.txt"
    path.write_bytes(b"\xffprefixTAIL")
    with path.open(encoding="utf-8") as source:
        source.seek(7)
        cookie = source.tell()
        with pytest.raises(UnicodeDecodeError):
            cloudpickle.dumps(source)
        assert source.tell() == cookie
