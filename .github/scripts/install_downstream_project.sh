python -m pip install $TEST_REQUIREMENTS
pushd ..
git clone $PROJECT_URL
pushd $PROJECT
git checkout 047b6eff2240c918cbc15282aad2241b6a0b612f 
popd
python -m pip install ./$PROJECT;
popd;
