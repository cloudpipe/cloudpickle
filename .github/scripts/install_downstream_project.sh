python -m pip install $TEST_REQUIREMENTS
pushd ..
git clone $PROJECT_URL
pushd $PROJECT
git checkout 53fcdbbcd9d3ba1bf96fb2075f92fb1925e2584d
popd
python -m pip install ./$PROJECT;
popd;
