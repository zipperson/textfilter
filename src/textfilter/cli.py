#!/usr/bin/env python3
import argparse
import re
import sys
import time
from typing import Dict, List, Tuple

from detoxify import Detoxify


# ============================================================
# ANSI Color Helpers
# ============================================================
def supports_color() -> bool:
    return sys.stdout.isatty()


class C:
    if supports_color():
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"

        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"

        BRIGHT_RED = "\033[91m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_BLUE = "\033[94m"
        BRIGHT_MAGENTA = "\033[95m"
        BRIGHT_CYAN = "\033[96m"
    else:
        RESET = BOLD = DIM = ""
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = ""
        BRIGHT_RED = BRIGHT_GREEN = BRIGHT_YELLOW = BRIGHT_BLUE = BRIGHT_MAGENTA = BRIGHT_CYAN = ""


# ============================================================
# UI Helpers
# ============================================================
def type_out(text: str, delay: float = 0.015, end: str = "\n") -> None:
    """Simple typing effect for CLI output."""
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print(end=end, flush=True)


def loading_bar(seconds: float = 1.5, width: int = 30, label: str = "Preparing") -> None:
    """Progress bar that runs for approximately `seconds`."""
    steps = max(1, width)
    step_time = seconds / steps
    for i in range(steps + 1):
        filled = int((i / steps) * width)
        bar = "█" * filled + " " * (width - filled)
        pct = int((i / steps) * 100)
        print(
            f"\r{C.CYAN}{label}{C.RESET}: [{C.BRIGHT_GREEN}{bar}{C.RESET}] {pct:3d}%",
            end="",
            flush=True,
        )
        time.sleep(step_time)
    print()


def show_banner() -> None:
    banner = rf"""{C.BRIGHT_MAGENTA}{C.BOLD}
██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗██╗████████╗██╗███████╗███████╗██████╗
██╔══██╗██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██║╚══██╔══╝██║╚══███╔╝██╔════╝██╔══██╗
██║  ██║█████╗  ███████╗█████╗  ██╔██╗ ██║███████╗██║   ██║   ██║  ███╔╝ █████╗  ██████╔╝
██║  ██║██╔══╝  ╚════██║██╔══╝  ██║╚██╗██║╚════██║██║   ██║   ██║ ███╔╝  ██╔══╝  ██╔══██╗
██████╔╝███████╗███████║███████╗██║ ╚████║███████║██║   ██║   ██║███████╗███████╗██║  ██║
╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝   ╚═╝   ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
{C.RESET}{C.DIM}========================================================{C.RESET}
"""
    print(banner)


def press_to_begin() -> bool:
    while True:
        print(f"{C.BOLD}Press 1 to begin{C.RESET}")
        print(f"{C.BOLD}Press 0 to exit{C.RESET}")
        choice = input("> ").strip()
        if choice == "1":
            return True
        if choice == "0":
            return False
        print(f"{C.YELLOW}Invalid input. Please enter 1 or 0.{C.RESET}\n")


def after_each_output_prompt() -> bool:
    print(f"\n{C.BOLD}Press 0 to stop, or press Enter to continue.{C.RESET}")
    return input("> ").strip() != "0"


# ============================================================
# Censor List Utilities
# ============================================================
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


# ============================================================
# AI per-word censoring (approximation)
# ============================================================
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


# ============================================================
# Main
# ============================================================
def format_scores(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(((k, float(v)) for k, v in scores.items()), key=lambda kv: kv[1], reverse=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Text Filtering CLI (list-based censor + AI per-word toxicity censor)"
    )
    p.add_argument("--terms", required=True, help="Path to terms.txt")
    p.add_argument("--mask", default="*", help="Mask character (default: *)")
    p.add_argument(
        "--model",
        default="original",
        choices=["original", "unbiased", "multilingual"],
        help="Detoxify model",
    )
    p.add_argument(
        "--ai-threshold",
        type=float,
        default=0.60,
        help="Per-word toxicity threshold for AI censoring",
    )
    args = p.parse_args()

    show_banner()
    type_out(f"{C.CYAN}Initializing...{C.RESET}", delay=0.01)

    if not press_to_begin():
        type_out(f"{C.YELLOW}Operation cancelled.{C.RESET}", delay=0.01)
        return

    # Load list terms first (fast)
    terms = load_terms(args.terms)
    censor_re = build_censor_regex(terms)

    # Progress before model load
    loading_bar(seconds=1.3, width=32, label="Loading resources")

    # Load model
    type_out(f"{C.CYAN}Loading Detoxify model: {args.model}...{C.RESET}", delay=0.01)
    model = Detoxify(args.model)
    type_out(f"{C.BRIGHT_GREEN}Model loaded. Ready.{C.RESET}\n", delay=0.01)

    # Configuration summary
    print(f"{C.BOLD}AI per-word threshold:{C.RESET} {C.BRIGHT_YELLOW}{args.ai_threshold}{C.RESET}")
    print(f"{C.BOLD}List terms loaded:{C.RESET} {C.BRIGHT_YELLOW}{len(terms)}{C.RESET}")

    while True:
        text = input(f"\n{C.BOLD}Enter text:{C.RESET} ").rstrip("\n")
        if not text.strip():
            continue

        # Always censor list terms first
        list_censored_text, list_matches = censor_list_terms(text, censor_re, args.mask)

        # Then censor toxic words (not already masked)
        final_censored, ai_hits = censor_toxic_words_ai(
            list_censored_text,
            model=model,
            ai_threshold=args.ai_threshold,
            mask_char=args.mask,
        )

        # Full-message scores for decisioning
        msg_scores = predict_safe(model, text)
        msg_tox = float(msg_scores.get("toxicity", 0.0))

        # Decision label
        if list_matches:
            decision = f"{C.BRIGHT_RED}{C.BOLD}BLOCKED (LIST){C.RESET}"
        elif msg_tox >= 0.80:
            decision = f"{C.BRIGHT_RED}BLOCK{C.RESET}"
        elif msg_tox >= 0.50:
            decision = f"{C.BRIGHT_YELLOW}REVIEW{C.RESET}"
        else:
            decision = f"{C.BRIGHT_GREEN}ALLOW{C.RESET}"

        # Output
        print(f"\n{C.BOLD}=== RESULT ==={C.RESET}")
        if list_matches:
            print(f"{C.BOLD}Decision:{C.RESET} {decision}")
        else:
            print(f"{C.BOLD}Decision:{C.RESET} {decision}  {C.DIM}(toxicity={msg_tox:.3f}){C.RESET}")

        print(f"{C.BOLD}Censored text:{C.RESET} {final_censored}")

        # Censor report
        print(f"\n{C.BOLD}CENSOR REPORT{C.RESET}")
        if not list_matches and not ai_hits:
            print(f"{C.DIM}- No words censored.{C.RESET}")
        else:
            if list_matches:
                seen = set()
                uniq = []
                for w in list_matches:
                    lw = w.lower()
                    if lw not in seen:
                        seen.add(lw)
                        uniq.append(w)
                for w in uniq:
                    print(f"- {C.BRIGHT_BLUE}'{w}'{C.RESET} censored by {C.BRIGHT_BLUE}[LIST]{C.RESET}")

            if ai_hits:
                seen = set()
                uniq_ai: List[Tuple[str, float]] = []
                for w, s in ai_hits:
                    lw = w.lower()
                    if lw not in seen:
                        seen.add(lw)
                        uniq_ai.append((w, s))
                for w, s in uniq_ai:
                    print(
                        f"- {C.BRIGHT_MAGENTA}'{w}'{C.RESET} censored by "
                        f"{C.BRIGHT_MAGENTA}[AI]{C.RESET} (word-tox={s:.3f})"
                    )

        # Show top message scores
        print(f"\n{C.BOLD}MESSAGE SCORES{C.RESET}")
        for k, v in format_scores(msg_scores)[:6]:
            print(f"  - {k:14s} {v:.3f}")

        if not after_each_output_prompt():
            type_out(f"{C.YELLOW}Stopping.{C.RESET}", delay=0.01)
            break


if __name__ == "__main__":
    main()
