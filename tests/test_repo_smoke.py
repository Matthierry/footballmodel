from pathlib import Path


def test_expected_project_files_exist():
    assert Path('pyproject.toml').exists()
    assert Path('src/footballmodel').exists()
    assert Path('.github/workflows/tests.yml').exists()
