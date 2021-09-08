pushd ../$PROJECT
python -m pytest -vl $ADDITIONAL_PYTEST_ARGS
TEST_RETURN_CODE=$?
popd
if [[ "$TEST_RETURN_CODE" != "0" ]]; then
    exit $TEST_RETURN_CODE
fi
