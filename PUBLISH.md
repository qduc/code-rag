# Publishing to PyPI Guide

This guide outlines the steps to publish the **Code-RAG** project to PyPI.

## Automated Release

You can use the provided release script to automate the version bumping, tagging, building, and publishing process:

```bash
python3 scripts/release.py
```

The script will:
1. Ask for the version bump type (patch, minor, major).
2. Update the version in `pyproject.toml` and `src/code_rag/__init__.py`.
3. Commit and tag the version in git.
4. Build the package using `python -m build`.
5. Upload the package to PyPI using `twine`.

---

## 1. Prerequisites
- Create an account on [PyPI](https://pypi.org/account/register/).
- (Optional but recommended) Create an account on [TestPyPI](https://test.pypi.org/account/register/) to test the upload first.
- Install the necessary tools:
  ```bash
  pip install --upgrade build twine
  ```

## 2. Configuration
We have already created a `pyproject.toml` file which contains the metadata and dependencies.
- **Action Required**: Open `pyproject.toml` and update the following fields:
  - `authors`: Replace "Your Name" and "your.email@example.com".
  - `project.urls`: Update the GitHub URLs to your repository.

## 3. Build the Package
Run the following command in the root directory:
```bash
python -m build
```
This will create a `dist/` directory containing the source distribution (`.tar.gz`) and the wheel (`.whl`).

## 4. Validate the Package
Check if the description and metadata are correct:
```bash
twine check dist/*
```

## 5. (Recommended) Upload to TestPyPI
It's a good idea to test the upload on the test server first:
```bash
twine upload --repository testpypi dist/*
```
You can then try installing it from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps code-rag
```

## 6. Upload to PyPI
Once everything looks good, upload to the real PyPI:
```bash
twine upload dist/*
```

## 7. Versioning
To publish a new version later:
1. Update the version in `pyproject.toml` and `src/code_rag/__init__.py`.
2. Delete the old `dist/` folder.
3. Repeat steps 3, 4, and 6.
