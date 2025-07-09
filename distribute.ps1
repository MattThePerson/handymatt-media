# Shell script for building and uploading to PyPi

$ErrorActionPreference = "Stop"

# Ensure required packages are installed and up to date
python -m pip install -U pip
python -m pip install -U build
python -m pip install -U twine
python -m pip install -U keyrings.alt

# Remove old build files and build
if (Test-Path dist) {
    Remove-Item -Recurse -Force dist\*
}
python -m build

# Upload
# python -m twine upload --repository testpypi dist\*
python -m twine upload dist\*
