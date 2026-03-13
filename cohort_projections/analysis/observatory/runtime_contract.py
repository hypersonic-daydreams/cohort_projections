"""Runtime parameter contract utilities for the Projection Observatory.

The Observatory needs to know which catalog parameters are actually injectable
through the live benchmark runtime. Rather than maintaining a second manual
list, this module reads the authoritative ``MethodConfig`` TypedDict from
``scripts/analysis/walk_forward_validation.py`` and exposes the parameter set
programmatically.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHOD_CONFIG_PATH = (
    PROJECT_ROOT / "scripts" / "analysis" / "walk_forward_validation.py"
)


@functools.lru_cache(maxsize=1)
def get_runtime_injectable_parameters(
    method_config_path: Path = DEFAULT_METHOD_CONFIG_PATH,
) -> frozenset[str]:
    """Return the injectable ``MethodConfig`` parameter names.

    Parameters
    ----------
    method_config_path
        Path to the canonical ``walk_forward_validation.py`` source file.

    Returns
    -------
    frozenset[str]
        The annotated field names on the ``MethodConfig`` TypedDict.

    Raises
    ------
    FileNotFoundError
        If the source file is missing.
    ValueError
        If the ``MethodConfig`` TypedDict cannot be located.
    """
    if not method_config_path.exists():
        raise FileNotFoundError(
            f"MethodConfig source not found: {method_config_path}"
        )

    tree = ast.parse(method_config_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "MethodConfig":
            continue
        fields = [
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
        ]
        if not fields:
            raise ValueError(
                "MethodConfig TypedDict was found but has no annotated fields."
            )
        return frozenset(fields)

    raise ValueError(
        f"MethodConfig TypedDict not found in {method_config_path}"
    )


def is_runtime_injectable(parameter: str) -> bool:
    """Return ``True`` if *parameter* is supported by the live runtime."""
    return parameter in get_runtime_injectable_parameters()
