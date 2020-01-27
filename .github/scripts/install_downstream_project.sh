python -m pip install $TEST_REQUIREMENTS
python -m pip uninstall -y cryptography
pushd ..
git clone $PROJECT_URL;
python -m pip install ./$PROJECT;
popd;
