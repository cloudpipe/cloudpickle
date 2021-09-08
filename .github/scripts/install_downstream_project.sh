python -m pip install $TEST_REQUIREMENTS
pushd ..
git clone $PROJECT_URL
pushd $PROJECT
git checkout HEAD~3
popd
python -m pip install ./$PROJECT;
popd;
