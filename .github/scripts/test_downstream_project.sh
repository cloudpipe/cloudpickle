pushd ../$PROJECT
echo "${DISABLE_IPV6}"
echo "${PYTEST_ADDOPTS[@]}"
python -m pytest -vl "${PYTEST_ADDOPTS[@]}"
TEST_RETURN_CODE=$?
popd
if [[ "$TEST_RETURN_CODE" != "0" ]]; then
    exit $TEST_RETURN_CODE
fi
