pushd ../$PROJECT
python -m pytest -vls -k remote_access
TEST_RETURN_CODE=$?
popd
if [[ "$TEST_RETURN_CODE" != "0" ]]; then
  exit $TEST_RETURN_CODE
fi
