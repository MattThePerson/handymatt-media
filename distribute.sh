# Shell script I use for building and uploading to PyPi

# make sure packages are installed and up to date
python3 -m pip install -U pip
python3 -m pip install -U build
python3 -m pip install -U twine
python3 -m pip install -U keyrings.alt

# remove old build files and build
rm -rf dist/*
python3 -m build


# upload
# python3 -m twine upload --repository testpypi dist/*
python3 -m twine upload dist/*

# END
