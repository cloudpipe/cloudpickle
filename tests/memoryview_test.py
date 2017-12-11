import sys
import platform
from cloudpickle.cloudpickle import _safe_to_mutate
from cloudpickle.cloudpickle import _memoryview_from_bytes
from cloudpickle.cloudpickle import _Py2StrStruct
import pytest


@pytest.mark.skipif(sys.version_info[:2] == (3, 4),
                    reason="https://bugs.python.org/issue19803")
def test_safe_mutable_bytes():
    if sys.version_info[0] >= 3:
        buffer = bytes([1, 2, 3])
    else:
        # Make the buffer large enough to avoid Python 2 string interning.
        buffer = (u'0123' * 100000).encode('ascii')
    buffer_id = id(buffer)
    buffer_size = len(buffer)
    buffer_holder = [buffer]
    # Delete the local direct reference to the buffer
    del buffer

    # There are no external reference to the bytes object in buffer_holder,
    # the _memoryview_from_bytes function can safely reuse the memory
    # allocated for the bytes object to expose it as a mutable buffer to back
    # the memoryview.
    assert _safe_to_mutate(buffer_holder)
    view = _memoryview_from_bytes(buffer_holder, 'B', False, (buffer_size,))
    assert not view.readonly
    assert len(buffer_holder) == 0

    # CPython 2 and PyPy do not expose the obj attribute.
    if hasattr(view, 'obj'):
        # The last reference to the original bytes object is hold in a closure
        # to discourage direct access by the user:
        with pytest.raises(TypeError) as exc_info:
            view.obj._hidden_buffer_ref()
            msg = "Access to mutable buffer with id %d is unsafe" % buffer_id
            assert exc_info.value.args[0] == msg

    # It's possible to mutate the content of the buffer
    view[:] = b'\x00' * buffer_size


def test_never_mutate_singleton_bytes():
    for i in range(256):
        if sys.version_info[0] >= 3:
            buffer_holder = [bytes([i])]
        else:
            buffer_holder = [chr(i)]
        assert not _safe_to_mutate(buffer_holder)

        # In this case, a new read-write buffer is allocated to back the
        # memoryview.
        view = _memoryview_from_bytes(buffer_holder, 'B', False, (1,))
        assert not view.readonly
        if hasattr(view, 'obj'):
            assert not hasattr(view.obj, '_hidden_buffer_ref')


def test_unsafe_mutable_bytes_with_external_references():
    buffer = u"\x01\x02\x03".encode('ascii')
    buffer_holder = [buffer]

    # The local 'buffer' variable still holds a reference to the bytes object
    # instance: _memoryview_from_bytes cannot safely reuse the same buffer.
    assert not _safe_to_mutate(buffer_holder)

    # Instead new mutable memory is allocated to back the buffer:
    view = _memoryview_from_bytes(buffer_holder, 'B', False, (3,))
    assert not view.readonly
    assert len(buffer_holder) == 0
    if hasattr(view, 'obj'):
        # CPython 2 and PyPy does not expose the 'obj attribute'.
        assert not hasattr(view.obj, '_hidden_buffer_ref')

    # Check that changing the content of the view does not change the content
    # of original buffer:
    view[:] = b'\x00\x00\x00'
    assert buffer == b"\x01\x02\x03"


@pytest.mark.skipif(sys.version_info[0] != 2,
                    reason="Test relies on Python 2 str interning")
@pytest.mark.skipif(platform.python_implementation != 'CPython',
                    reason="Test requires access to ob_sstate field.")
def test_py2_interned_string_detection():
    def process_str(original):
        """Return non-interned version of original str object"""
        return ' '.join(x for x in original.split(' '))
    s1 = process_str('some python 2 str')
    s2 = intern('some python 2 str')
    s3 = intern('some python 2 str')
    assert s1 == s2
    assert s1 is not s2
    assert s2 is s3
    assert not _Py2StrStruct.from_address(id(s1)).is_interned()
    assert _Py2StrStruct.from_address(id(s2)).is_interned()
    assert _Py2StrStruct.from_address(id(s3)).is_interned()

    # Interned CPython str buffers are never safe to mutate even without
    # external references.
    buffer_holder = [intern(process_str('other python 2 str'))]
    assert not _safe_to_mutate(buffer_holder)

    # Non-interned CPython str buffers without any external references are
    # safe to mutate:
    buffer_holder = [process_str('other python 2 str')]
    assert _safe_to_mutate(buffer_holder)


@pytest.mark.skipif(sys.version_info[0] != 2,
                    reason="Test relies on Python 2 str interning")
def test_unsafe_mutable_bytes_with_python_2_interning():
    buffer_holder = [intern(u"\x01\x02\x03".encode('ascii'))]

    assert not _safe_to_mutate(buffer_holder)
    view = _memoryview_from_bytes(buffer_holder, 'B', False, (3,))
    assert not view.readonly
    assert len(buffer_holder) == 0

    # Create a new interned buffer with the same original content
    interned_buffer = intern(b"\x01\x02\x03")

    # Mutate the view pointing the to the original buffer
    view[:] = b'\x00\x00\x00'
    assert interned_buffer == b'\x01\x02\x03'
