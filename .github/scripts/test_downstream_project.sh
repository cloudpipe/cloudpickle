pushd ../$PROJECT
python -m pytest -vl
TEST_RETURN_CODE=$?
popd
if [[ "$TEST_RETURN_CODE" != "0" ]]; then
    exit $TEST_RETURN_CODE
fi
