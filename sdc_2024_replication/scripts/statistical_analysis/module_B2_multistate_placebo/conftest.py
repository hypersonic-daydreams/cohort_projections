# This conftest.py prevents pytest from collecting this directory as tests.
# The test_* aliases in __init__.py are for API compatibility, not pytest tests.

collect_ignore_glob = ["*.py"]
