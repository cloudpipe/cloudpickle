import sys
import platform
from cloudpickle.cloudpickle import _is_safe_to_mutate
from cloudpickle.cloudpickle import _memoryview_from_bytes
from cloudpickle.cloudpickle import _Py2StrStruct
import pytest


RUNNING_CPYTHON = platform.python_implementation() == 'CPython'


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
    # the memoryview when running CPython.
    if RUNNING_CPYTHON:
        assert _is_safe_to_mutate(buffer_holder)
    else:
        assert not _is_safe_to_mutate(buffer_holder)
    view = _memoryview_from_bytes(buffer_holder, 'B', False, (buffer_size,))
    assert not view.readonly
    assert len(buffer_holder) == 0

    # CPython 2 and PyPy do not expose the obj attribute.
    if hasattr(view, 'obj') and RUNNING_CPYTHON:
        # The last reference to the original bytes object is hold in a closure
        # to discourage direct access by the user:
        with pytest.raises(TypeError) as exc_info:
            view.obj._hidden_buffer_ref()
            msg = "Access to mutable buffer with id %d is unsafe" % buffer_id
            assert exc_info.value.args[0] == msg

    # It's possible to mutate the content of the buffer
    view[:] = b'\x00' * buffer_size


@pytest.mark.skipif(sys.version_info[0] < 3 or not RUNNING_CPYTHON,
                    reason="Test relies on CPython 3 implementation details")
def test_mutate_py3_single_element_bytes():
    """Check the safety of mutating a single element bytes instance.

    The _memoryview_from_bytes relies on implementation details of the
    CPython 3 interpreter when deciding whether it is safe to mutate the
    existing buffer or not.

    In particular it assumes that when a singleton bytes instance is implicitly
    interned, it has always at least one external reference somewhere else in
    the interpreter and the _safe_to_mutate will therefore always return False.

    If this assumption is ever violated in some future version of CPython,
    this test will fail and will tell the cloudpickle maintainers that safety
    of mutating singleton bytes is no longer guaranteed.

    This test is therefore a canary test for future Python 3 versions
    """
    # Single element bytes can be singleton (implicitly interned), for instance
    # when they are sliced from a larger bytes object or when encoding a
    # single symbol unicode string.
    joined_bytes = bytes(range(256))
    implicitly_interned_bytes = []
    for i in range(256):
        assert sys.getrefcount(joined_bytes[i:i + 1]) >= 2
        buffer_holder = [joined_bytes[i:i + 1]]
        assert not _is_safe_to_mutate(buffer_holder)
        implicitly_interned_bytes.append(buffer_holder[0])
        assert chr(i).encode('latin1') is implicitly_interned_bytes[i]
        assert joined_bytes[i:i + 1] is implicitly_interned_bytes[i]

        # In this case, a new read-write buffer is allocated to back the
        # memoryview.
        view = _memoryview_from_bytes(buffer_holder, 'B', False, (1,))
        assert not view.readonly
        if sys.version_info[:2] != (3, 4):
            assert not hasattr(view.obj, '_hidden_buffer_ref')

    # Single item bytes instances are not interned when using the bytes([i])
    # constructor and this case, they can be mutated safely, they will not
    # impact the implicitly interned bytes singleton instances.
    for i in range(256):
        buffer_holder = [bytes([i])]
        assert buffer_holder[0] == implicitly_interned_bytes[i]
        assert buffer_holder[0] is not implicitly_interned_bytes[i]
        assert _is_safe_to_mutate(buffer_holder)
        view = _memoryview_from_bytes(buffer_holder, 'B', False, (1,))
        if sys.version_info[:2] != (3, 4):
            with pytest.raises(TypeError):
                view.obj._hidden_buffer_ref()
        view[:] = b'\x00'
        assert view.tobytes() == b'\x00'

    # The singleton are not impacted by the changes in the views.
    for i in range(256):
        assert implicitly_interned_bytes[i] == bytes([i])
        assert implicitly_interned_bytes[i] is not bytes([i])


@pytest.mark.skipif(sys.version_info[0] < 3,
                    reason="Test checks assumption on Python 3 implementation")
def test_check_no_explicit_py3_bytes_interning():
    """Ensure that Python 3+ does not allow explicit interning for bytes.

    The _memoryview_from_bytes function makes the assumption that it is not
    possible to explicitly intern bytes objects. If this assumption is ever
    violated in future versions of Python, the implementation of
    _memoryview_from_bytes will have to be updated accordingly to preserve the
    safety of directly reusing a bytes buffer to a mutable memoryview.

    This is a canary test: this does not actually test anything in cloudpickle
    besides assumptions made on the language implementation.
    """
    with pytest.raises(TypeError):
        sys.intern(b'some bytes instance')


def test_unsafe_mutable_bytes_with_external_references():
    buffer = u"\x01\x02\x03".encode('ascii')
    buffer_holder = [buffer]

    # The local 'buffer' variable still holds a reference to the bytes object
    # instance: _memoryview_from_bytes cannot safely reuse the same buffer.
    assert not _is_safe_to_mutate(buffer_holder)

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
    assert not _is_safe_to_mutate(buffer_holder)

    # Non-interned CPython str buffers without any external references are
    # safe to mutate:
    buffer_holder = [process_str('other python 2 str')]
    assert _is_safe_to_mutate(buffer_holder)


@pytest.mark.skipif(sys.version_info[0] != 2,
                    reason="Test relies on Python 2 str interning")
def test_unsafe_mutable_bytes_with_python_2_interning():
    buffer_holder = [intern(u"\x01\x02\x03".encode('ascii'))]

    assert not _is_safe_to_mutate(buffer_holder)
    view = _memoryview_from_bytes(buffer_holder, 'B', False, (3,))
    assert not view.readonly
    assert len(buffer_holder) == 0

    # Create a new interned buffer with the same original content
    interned_buffer = intern(b"\x01\x02\x03")

    # Mutate the view pointing the to the original buffer
    view[:] = b'\x00\x00\x00'
    assert interned_buffer == b'\x01\x02\x03'
