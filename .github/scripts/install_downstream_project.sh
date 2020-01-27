python -m pip install $TEST_REQUIREMENTS
python -m pip install -U certifi
pushd ..
git clone $PROJECT_URL;
python -m pip install ./$PROJECT;
popd;
