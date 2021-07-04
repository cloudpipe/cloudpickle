# simulate an external function which cloudpickle would normally
# only save a reference to
mutable_variable = ["original_string"]
mutable_variable2 = ["_second_string"]


def inner_function():
    return mutable_variable[0]


def wrapping_func():
    from .external_two import function_from_external_two
    return (
        inner_function() + mutable_variable2[0] +
        function_from_external_two()
    )
