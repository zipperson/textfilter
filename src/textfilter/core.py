import re
from typing import Dict, List, Tuple

from detoxify import Detoxify


def load_terms(path: str) -> List[str]:
    terms: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                terms.append(t)
    return terms


def build_censor_regex(terms: List[str]) -> re.Pattern:
    """
    Build a regex that matches any term in the list.
    - Single words use word boundaries to reduce false positives.
    - Phrases match as-is.
    """
    cleaned = [t.strip() for t in terms if t.strip()]
    if not cleaned:
        return re.compile(r"(?!x)x")  # match nothing

    escaped = [re.escape(t) for t in cleaned]
    patterns: List[str] = []

    for t, e in zip(cleaned, escaped):
        patterns.append(e if " " in t else rf"\b{e}\b")

    return re.compile("|".join(patterns), flags=re.IGNORECASE)


def mask_word(word: str, mask_char: str = "*") -> str:
    return mask_char * len(word)


def censor_list_terms(text: str, censor_re: re.Pattern, mask_char: str) -> Tuple[str, List[str]]:
    """
    Censor all list terms immediately.
    Returns (censored_text, list_matches_original_forms).
    """
    matches = [m.group(0) for m in censor_re.finditer(text)]
    censored = censor_re.sub(lambda m: mask_word(m.group(0), mask_char), text)
    return censored, matches


def split_tokens(text: str) -> List[str]:
    """Split text into word and non-word tokens while preserving punctuation/spaces."""
    return re.findall(r"\w+|\W+", text)


def is_word_token(tok: str) -> bool:
    return tok.strip().isalnum()


def predict_safe(model: Detoxify, text: str) -> Dict[str, float]:
    return model.predict(text)


def censor_toxic_words_ai(
    text: str,
    model: Detoxify,
    ai_threshold: float,
    mask_char: str,
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Censor only words whose per-word toxicity >= ai_threshold.
    Returns (censored_text, ai_censored_words_with_scores).
    """
    tokens = split_tokens(text)
    ai_hits: List[Tuple[str, float]] = []
    out: List[str] = []

    for tok in tokens:
        if is_word_token(tok):
            # If already masked (e.g., ******), do not score it again.
            if set(tok) == {mask_char}:
                out.append(tok)
                continue

            score = float(predict_safe(model, tok).get("toxicity", 0.0))
            if score >= ai_threshold:
                out.append(mask_word(tok, mask_char))
                ai_hits.append((tok, score))
            else:
                out.append(tok)
        else:
            out.append(tok)

    return "".join(out), ai_hits


def format_scores(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(((k, float(v)) for k, v in scores.items()), key=lambda kv: kv[1], reverse=True)


class TextFilter:
    """
    Reusable class wrapper:
    - loads a term list
    - loads Detoxify model
    - processes text into a structured result
    """

    def __init__(
        self,
        terms_path: str,
        model_name: str = "original",
        mask_char: str = "*",
        ai_threshold: float = 0.60,
    ):
        self.terms_path = terms_path
        self.model_name = model_name
        self.mask_char = mask_char
        self.ai_threshold = ai_threshold

        self.terms = load_terms(terms_path)
        self.censor_re = build_censor_regex(self.terms)
        self.model = Detoxify(model_name)

    def process(self, text: str) -> Dict[str, object]:
        """
        Returns a dictionary result that can be used by CLI, web apps, automations, etc.
        """
        # Censor list terms first
        list_censored_text, list_matches = censor_list_terms(text, self.censor_re, self.mask_char)

        # Then censor toxic words (not already masked)
        final_censored, ai_hits = censor_toxic_words_ai(
            list_censored_text,
            model=self.model,
            ai_threshold=self.ai_threshold,
            mask_char=self.mask_char,
        )

        # Full-message scores for decisioning
        msg_scores = predict_safe(self.model, text)
        msg_tox = float(msg_scores.get("toxicity", 0.0))

        # Decision label
        if list_matches:
            decision = "BLOCKED_LIST"
        elif msg_tox >= 0.80:
            decision = "BLOCK"
        elif msg_tox >= 0.50:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        return {
            "decision": decision,
            "toxicity": msg_tox,
            "censored_text": final_censored,
            "list_matches": list_matches,
            "ai_hits": ai_hits,
            "scores": msg_scores,
        }
