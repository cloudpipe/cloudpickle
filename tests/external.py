# simulate an external function which cloudpickle would normally
# only save a reference to
mutable_variable = ["return_a_string"]
mutable_variable2 = ["_nested"]

def an_external_function():
    return mutable_variable[0]

def nested_function():
    return an_external_function() + mutable_variable2[0]