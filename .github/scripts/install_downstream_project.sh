python -m pip install $TEST_REQUIREMENTS
pushd ..
git clone $PROJECT_URL;
python -m pip install ./$PROJECT;
popd;
