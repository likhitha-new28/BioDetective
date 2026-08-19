from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_required_github_sections():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    required_sections = (
        "Project purpose",
        "Problem statement",
        "Screenshots",
        "Features",
        "Installation",
        "Quick start",
        "Expected data format",
        "Architecture",
        "Analysis methods",
        "Synthetic benchmark",
        "Limitations",
        "Scientific disclaimer",
    )

    assert all(f"## {section}" in readme for section in required_sections)


def test_repository_policy_files_exist_and_gitignore_covers_sensitive_artifacts():
    for filename in ("LICENSE", "CONTRIBUTING.md", "CHANGELOG.md"):
        assert (PROJECT_ROOT / filename).is_file()

    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("__pycache__/", ".pytest_cache/", ".env", ".streamlit/secrets.toml", "uploads/", "datasets/"):
        assert pattern in gitignore
