python -m pip install $TEST_REQUIREMENTS
pushd ..
git clone $PROJECT_URL;
pushd $PROJECT
git checkout fixup_tst_pickle
popd
python -m pip install ./$PROJECT;
popd;
