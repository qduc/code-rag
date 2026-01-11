import os
import re
import subprocess
import sys


def get_current_version():
    with open("pyproject.toml", "r") as f:
        content = f.read()
        # Look for version specifically under [project]
        match = re.search(r'\[project\][\s\S]*?\nversion\s*=\s*"(.*?)"', content)
        if match:
            return match.group(1)
    return None


def update_version(new_version):
    # Update pyproject.toml
    with open("pyproject.toml", "r") as f:
        content = f.read()

    # Precise replacement for version under [project]
    new_content = re.sub(
        r'(\[project\][\s\S]*?\nversion\s*=\s*")([^"]*)(")',
        rf"\g<1>{new_version}\g<3>",
        content,
        count=1,
    )

    with open("pyproject.toml", "w") as f:
        f.write(new_content)


def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)


def validate_bump(bump, current_version):
    parts = current_version.split(".")
    if len(parts) == 3:
        major, minor, patch = map(int, parts)
        if bump in ["patch", "p"]:
            return f"{major}.{minor}.{patch + 1}"
        elif bump in ["minor", "m"]:
            return f"{major}.{minor + 1}.0"
        elif bump in ["major", "M"]:
            return f"{major + 1}.0.0"
    if re.match(r"^\d+\.\d+\.\d+$", bump):
        return bump
    return None


def main():
    # Ensure we are in the root directory
    if not os.path.exists("pyproject.toml"):
        print(
            "Error: pyproject.toml not found. Please run this script from the project root."
        )
        sys.exit(1)

    current_version = get_current_version()
    if not current_version:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)

    print(f"Current version: {current_version}")

    bump = None
    if len(sys.argv) > 1:
        bump = sys.argv[1].strip().lower()

    if bump:
        new_version = validate_bump(bump, current_version)
        if not new_version:
            print(
                "Invalid bump type. Use patch/p, minor/m, major/M, or a version like 1.2.3"
            )
            sys.exit(1)
    else:
        while True:
            bump_input = (
                input("Bump version? (patch/p/minor/m/major/M) or enter new version: ")
                .strip()
                .lower()
            )
            if not bump_input:
                continue
            new_version = validate_bump(bump_input, current_version)
            if new_version:
                break
            else:
                print(
                    "Invalid input. Please enter 'patch', 'p', 'minor', 'm', 'major', 'M' or a version like '1.2.3'."
                )

    print(f"Bumping to version: {new_version}")

    confirm = input("Proceed? (Y/n): ").strip().lower()
    if confirm != "y" and confirm != "":
        print("Aborted.")
        sys.exit(0)

    # 1. Update version in files
    update_version(new_version)
    print("Updated version in pyproject.toml and src/code_rag/__init__.py")

    # 2. Commit and Tag
    files_to_add = ["pyproject.toml"]
    if os.path.exists("src/code_rag/__init__.py"):
        files_to_add.append("src/code_rag/__init__.py")

    run_command(f"git add {' '.join(files_to_add)}")
    run_command(f'git commit -m "chore: release version {new_version}"')
    run_command(f'git tag -a v{new_version} -m "version {new_version}"')
    print(f"Committed and tagged v{new_version}")

    # 3. Push the changes
    run_command("git push")
    run_command("git push --tags")
    print(f"Pushed changes and tag v{new_version}")

    # 4. Build
    print("Building package...")
    if os.path.exists("dist"):
        import shutil

        shutil.rmtree("dist")
    if os.path.exists("build"):
        import shutil

        shutil.rmtree("build")
    run_command("python3 -m build")

    # 4. Upload to PyPI
    print("Uploading to PyPI...")
    run_command("python3 -m twine upload dist/*")

    print("\nRelease successful!")


if __name__ == "__main__":
    main()
