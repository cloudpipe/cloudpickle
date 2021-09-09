pushd ../$PROJECT
python -m pytest -vl $ADDITIONAL_PYTEST_ARGS -k "pickle or serializ"
TEST_RETURN_CODE=$?
popd
if [[ "$TEST_RETURN_CODE" != "0" ]]; then
    exit $TEST_RETURN_CODE
fi
