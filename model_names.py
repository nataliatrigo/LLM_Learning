"""Public model names; historical data identifiers stay stable after relocation."""

import re


MODEL_LABELS = {'oe': 'OE distribution', 'p0': 'OE p0'}

_LEGACY_NAMES = (
    (r'OE\s+mixto\s+con\s+(?:la\s+)?marginal\s+del\s+juego\s+(?:conjunto\s+)?real', 'oe'),
    (r'OE\s*\(real[- ]game\s+marginal\)', 'oe'),
    (r'OE\s+with\s+real\s+marginal', 'oe'),
    (r'OE\s+(?:con|de)\s+marginal\s+real', 'oe'),
    (r'OE\s+real\s+marginal', 'oe'),
    (r'Endogenous[ -]+(?:p[0₀]|\$p_0\$)', 'p0'),
    (r'p[0₀]\s+end[oó]geno', 'p0'),
)


def rename_display_text(text, *, expand_short_oe=False):
    """Rename human-facing prose without changing internal underscore IDs."""
    for pattern, key in _LEGACY_NAMES:
        text = re.sub(pattern, MODEL_LABELS[key], text, flags=re.IGNORECASE)
    if expand_short_oe:
        text = re.sub(r'\bOE\b(?!\s+(?:distribution|p0)\b)', MODEL_LABELS['oe'], text)
    return text
