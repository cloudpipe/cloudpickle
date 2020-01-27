$PYTHON_EXE -m pip install $TEST_REQUIREMENTS
pushd ..
git clone $PROJECT_URL;
$PYTHON_EXE -m pip install ./$PROJECT;
popd;
