python -m pip install $TEST_REQUIREMENTS
python -m pip uninstall -y cryptography
pushd ..
git clone $PROJECT_URL;

pushd $PROJECT
git checkout debug-test-remote-access
popd

python -m pip install ./$PROJECT;
popd;
