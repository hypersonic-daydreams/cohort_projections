import os
import re

PROJECT_ROOT = os.getcwd()


def update_links():
    print("Updating links in markdown files...")

    # Mappings for absolute-ish paths (text replacement)
    # Be careful with partial matches.
    replacements = [
        ("docs/governance/adrs", "docs/governance/adrs"),
        ("docs/governance/sops", "docs/governance/sops"),
        ("docs/governance/reports", "docs/governance/reports"),
        (
            "docs/governance/templates/",
            "docs/governance/docs/governance/templates/",
        ),  # Context dependent?
    ]

    # Walk all files
    for root, _dirs, files in os.walk(PROJECT_ROOT):
        if ".git" in root or ".venv" in root:
            continue

        for file in files:
            if not file.endswith(".md") and not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path) as f:
                    content = f.read()

                original_content = content

                # 1. Update explicit paths
                for old, new in replacements:
                    content = content.replace(old, new)

                # 2. Update relative links if this file is INSIDE the moved directories
                # docs/governance/adrs (depth 2) -> docs/governance/adrs (depth 3)
                # docs/governance/sops (depth 2) -> docs/governance/sops (depth 3)
                # So if file is in `docs/governance/adrs`, and has `../`, it usually meant `docs/`.
                # Now `../` means `docs/governance/`. So we need `../../` to get to `docs/`.

                # Heuristic: If we are in docs/governance/*, we are 1 level deeper than before.
                # So replace `../` with `../../` ONLY IF it's referencing something outside governance?
                # Actually, `../` in docs/governance/adrs meant `docs/`.
                # `../` in docs/governance/adrs means `docs/governance`.
                # So `../` needs to become `../../` to preserve meaning "parent of my set".

                # But wait, if it was `../img.png`, it now needs to be `../../img.png` if img didn't move.
                # If `img.png` also moved (e.g. adr/figures), then relative link `figures/img.png` stays same.

                # Let's handle the specific case of `../` prefixes

                rel_path = os.path.relpath(file_path, PROJECT_ROOT)

                # Check if this file was one of the moved ones
                # It is now in docs/governance/adrs or sops or reports
                is_moved_file = (
                    "docs/governance/adrs" in rel_path
                    or "docs/governance/sops" in rel_path
                    or "docs/governance/reports" in rel_path
                )

                if is_moved_file:
                    # Replace `../` with `../../` for links that go UP
                    # But NOT if they are strictly internal like `../020-reports/` if that logic holds?
                    # No, usually `../` means escaping.

                    # Regex to find markdown links `[text](../path)`
                    # We want to change `(../` to `(../../`
                    # But only if it's not `(../../` already (from previous run) - wait we just moved.

                    # Simple Approach:
                    # If line has `(../`, replace with `(../../`

                    # But we must be careful not to break `(../../` if there was one.
                    # e.g. `(../../config)` becomes `(../../../config)`.

                    # Regex lookbehind or just simple substitution
                    # Find `](..` or `src="..`

                    def adjust_relative(match):
                        # match group 1 is the prefix `](` or `src="`
                        # match group 2 the dots `..`
                        # match group 3 the rest
                        return f"{match.group(1)}../{match.group(2)}{match.group(3)}"

                    # Matches `](..` or `src="..` or `href="..`
                    # Note: We are inserting an extra `../`
                    content = re.sub(r'(\]\(|src=["\']|href=["\'])\.\./', r"\1../../", content)

                if content != original_content:
                    print(f"Updating {rel_path}...")
                    with open(file_path, "w") as f:
                        f.write(content)

            except Exception as e:
                print(f"Skipping {file}: {e}")


if __name__ == "__main__":
    update_links()
