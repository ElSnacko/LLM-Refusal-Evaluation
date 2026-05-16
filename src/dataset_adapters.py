"""Dataset adapters for normalizing HuggingFace datasets into the pipeline's expected format.

Each adapter provides column mapping defaults (prompt_column, category_column) that are
applied when the user has not explicitly configured these values. Adapters are registered
by dataset ID prefix and auto-selected during loading.
"""

from typing import Dict, Optional

# Registry of known dataset adapters.
# Keys are matched as prefixes against dataset_id (case-insensitive).
# Values are dicts with column mapping defaults.
KNOWN_ADAPTERS: Dict[str, Dict[str, str]] = {
    "PKU-Alignment/BeaverTails": {
        "prompt_column": "prompt",
        "category_column": "auto",
    },
    "allenai/wildjailbreak": {
        "prompt_column": "vanilla",
        "category_column": "risk_category",
    },
    "sorry-bench": {
        "prompt_column": "prompt",
        "category_column": "category",
    },
}


def get_adapter_defaults(dataset_id: str) -> Optional[Dict[str, str]]:
    """Look up adapter defaults for a dataset ID.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., "PKU-Alignment/BeaverTails").

    Returns:
        Dict with "prompt_column" and "category_column" keys, or None if no adapter found.
    """
    dataset_id_lower = dataset_id.lower()
    for prefix, defaults in KNOWN_ADAPTERS.items():
        prefix_lower = prefix.lower()
        if dataset_id_lower.startswith(prefix_lower):
            # Ensure the match is not just a partial prefix match of a different dataset
            # e.g., "PKU-something-else" should NOT match "PKU-Alignment/BeaverTails"
            # After the prefix, either we've reached the end or the next char is a separator
            if len(dataset_id_lower) == len(prefix_lower):
                # Exact match (case-insensitive)
                return dict(defaults)
            # Check if the next character after the prefix is a path separator
            # or if the dataset_id contains the prefix as a full component
            if dataset_id_lower[len(prefix_lower):].startswith(('/', '-')):
                return dict(defaults)
    return None
