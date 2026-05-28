from __future__ import annotations

import re
from typing import List


class DynamicPromptBuilder:
    BASEPROMPT = (
        "Sei un assistente RAG per Smallthing Studios.\n"
        "Rispondi sempre in Italiano e usa SOLO le informazioni dai documenti forniti.\n"
    )

    CODE_PATTERN = re.compile(r"\b([A-Z]{2,5})(\d{2,4})\b")

    def __init__(self, max_examples: int = 5):
        self.max_examples = max_examples

    def _extract_codes_and_prefixes(self, texts: List[str]):
        prefixes = set()
        codes = []
        for t in texts:
            for m in self.CODE_PATTERN.finditer(t or ""):
                prefixes.add(m.group(1))
                codes.append(m.group(1) + m.group(2))
        return sorted(prefixes), list(dict.fromkeys(codes))

    def build(self, context_chunks: List[dict]) -> str:
        texts = [c.get("text", "") for c in context_chunks or []]
        prefixes, codes = self._extract_codes_and_prefixes(texts)

        dynamic_parts = []
        if prefixes:
            dynamic_parts.append("CODICI RICONOSCIUTI: " + ", ".join(prefixes))
        if codes:
            sample_codes = codes[: self.max_examples]
            dynamic_parts.append("ESEMPI DAI DOCUMENTI: " + ", ".join(sample_codes))

        # Build a compact context summary that the SessionManager can append if desired
        context_summaries = []
        for c in context_chunks or []:
            match_info = "✓ MATCH ESATTO" if c.get("match_type") == "keyword" else f"similarità: {c.get('similarity', 'N/A')}"
            context_summaries.append(f"--- Fonte: {c.get('source','unknown')} ({match_info}) ---\n{c.get('text','')}")

        parts = [self.BASEPROMPT]
        if dynamic_parts:
            parts.append("\n".join(dynamic_parts))
        if context_summaries:
            parts.append("\n\nDOCUMENTI INTERNI:\n\n" + "\n\n".join(context_summaries))

        parts.append("\nIMPORTANTE: Rispondi basandoti ESCLUSIVAMENTE sui documenti sopra.")

        return "\n\n".join(parts)
