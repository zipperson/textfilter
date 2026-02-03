#!/usr/bin/env python3
import argparse
import re
import sys
import time
from typing import Dict, List, Tuple, Optional

from detoxify import Detoxify


# ============================================================
# ANSI Color Helpers (works great in WSL / Windows Terminal)
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
        WHITE = "\033[37m"

        BRIGHT_RED = "\033[91m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_BLUE = "\033[94m"
        BRIGHT_MAGENTA = "\033[95m"
        BRIGHT_CYAN = "\033[96m"
    else:
        RESET = BOLD = DIM = ""
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
        BRIGHT_RED = BRIGHT_GREEN = BRIGHT_YELLOW = BRIGHT_BLUE = BRIGHT_MAGENTA = BRIGHT_CYAN = ""


# ============================================================
# UI Effects
# ============================================================
def type_out(text: str, delay: float = 0.02, end: str = "\n") -> None:
    """Typing animation."""
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print(end=end, flush=True)

def loading_bar(seconds: float = 2.0, width: int = 30, label: str = "Preparing philippine national police") -> None:
    """Progress bar that runs for approx `seconds`."""
    steps = max(1, width)
    step_time = seconds / steps
    for i in range(steps + 1):
        filled = int((i / steps) * width)
        bar = "█" * filled + " " * (width - filled)
        pct = int((i / steps) * 100)
        print(f"\r{C.CYAN}{label}{C.RESET}: [{C.BRIGHT_GREEN}{bar}{C.RESET}] {pct:3d}%", end="", flush=True)
        time.sleep(step_time)
    print()

def show_big_banner() -> None:
    banner = rf"""{C.BRIGHT_MAGENTA}{C.BOLD}
██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗██╗████████╗██╗███████╗███████╗██████╗ 
██╔══██╗██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██║╚══██╔══╝██║╚══███╔╝██╔════╝██╔══██╗
██║  ██║█████╗  ███████╗█████╗  ██╔██╗ ██║███████╗██║   ██║   ██║  ███╔╝ █████╗  ██████╔╝
██║  ██║██╔══╝  ╚════██║██╔══╝  ██║╚██╗██║╚════██║██║   ██║   ██║ ███╔╝  ██╔══╝  ██╔══██╗
██████╔╝███████╗███████║███████╗██║ ╚████║███████║██║   ██║   ██║███████╗███████╗██║  ██║
╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝   ╚═╝   ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
{C.RESET}{C.BRIGHT_YELLOW}{C.BOLD}                 pra kay LEBRON JAMES{C.RESET}
{C.DIM}========================================================{C.RESET}
"""
    print(banner)

def show_lebron_ascii() -> None:
    """
    sino yan kung ende si leBRON james ng ating bansa
    """
    art = rf"""{C.BRIGHT_YELLOW}{C.BOLD}
    
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣶⣾⣿⣿⣿⣶⣶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⠛⠛⠉⠉⠉⠉⠛⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⡿⠟⠀⠀⠀⠀⠀⠀⠀⠺⠶⣾⣿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡿⣇⠀⢀⡀⠀⠀⠀⠀⢀⣀⣀⣌⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⢣⣿⠆⢀⡀⠀⠀⠀⠀⠈⠉⢉⡺⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣿⣿⣾⡿⣻⣿⣿⡷⠇⠀⠀⣴⣿⣿⣿⣿⢿⣿⣿⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣟⣿⣿⡏⣾⣿⣿⣯⣿⣶⠾⠻⣿⣿⣽⣿⣿⣿⣿⣿⢿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  {C.BRIGHT_CYAN}{C.BOLD}LEBRON JAMES???{C.RESET}{C.BRIGHT_YELLOW}{C.BOLD}
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣻⣿⣿⢁⣙⡋⠉⠉⣿⡇⠀⠀⣺⣿⡯⠻⠛⠛⣿⣿⢉⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  {C.DIM}(walang iba, kundi si leBRON james ng PILIPINS){C.RESET}{C.BRIGHT_YELLOW}{C.BOLD}
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⡽⡏⠁⠀⣴⢿⣏⣥⣄⣠⣤⣽⡿⠆⠀⠀⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣇⣿⢰⣿⣤⣶⣾⡿⢿⡿⢿⣿⣶⣦⣌⢢⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣄⣻⣿⣿⣯⣉⡉⣉⣉⣩⣿⣿⣿⣼⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⡋⠠⣤⣤⣤⣴⡾⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣦⡀⠿⠃⣀⣴⣿⣿⣿⣿⣿⣿⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣀⡀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣼⠏⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠉⣿⣿⣿⣿⠛⡗⠶⢤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⣾⠻⣿⡟⣿⠀⠀⠀⠀⠈⠛⢉⣠⡝⠋⠉⠀⠀⠀⢻⣯⣿⠇⣼⢃⡆⠀⠈⠙⡶⢤⢤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠞⠋⢳⡘⣧⡹⣿⡹⡇⠀⠀⠀⠀⠀⠚⠛⠀⠀⠀⠀⠀⠀⣼⣿⠏⣰⠋⡜⠀⠀⠀⠀⡇⢸⢸⣷⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⢻⠛⡇⠀⠀⠀⠳⣌⢷⡌⢷⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⠏⣴⠋⠐⠁⡀⠀⠀⠀⡇⢸⢸⣿⢹⣯⣻⣷⣦⣤⡀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣀⣴⣾⣿⠘⡇⢹⠀⠀⠀⠀⠙⢦⡙⢦⣍⡻⠶⣤⣀⣀⠀⠀⠀⣀⣴⣾⡿⢛⣥⠞⡡⠀⠰⣿⡿⣿⡆⠀⠀⢸⠈⣯⠲⣏⠉⢻⣾⣿⡿⣦⡀⠀⠀
⠀⠀⠀⠀⣀⣴⣾⣿⣿⠟⢻⠀⣷⢸⠀⠀⠀⠀⠀⠀⠈⠓⢮⣙⠳⠦⣌⣙⣛⣳⣞⣛⣋⡥⠶⠋⠥⠛⠀⢠⣤⣭⣟⡟⣀⣀⡀⢸⡄⢿⠀⠀⠀⠈⢹⣿⡀⠬⣿⣄⠀
⢀⣠⣶⣿⣿⣿⡉⠉⠀⠀⢸⠀⣿⠸⠀⠀⠀⠀⠀⠀⢀⡤⠄⠈⠙⢒⢦⢀⢈⣉⡉⡁⠀⢀⠀⠀⠀⠀⠀⠈⠛⠛⠛⠻⠿⡛⠛⠈⡇⢸⠀⠀⠀⠀⢸⡟⠳⢆⢈⣿⣄

{C.RESET}"""
    print(art)

def press_to_begin() -> bool:
    while True:
        print(f"{C.BOLD}Press 1 to begin desensitizing{C.RESET}")
        print(f"{C.BOLD}Press 0 to exit{C.RESET}")
        choice = input("> ").strip()
        if choice == "1":
            return True
        if choice == "0":
            return False
        print(f"{C.YELLOW}Invalid input. Please press 1 or 0.{C.RESET}\n")

def after_each_output_prompt() -> bool:
    print(f"\n{C.BOLD}Press 0 to stop desensitizing{C.RESET}, or press Enter to continue.")
    choice = input("> ").strip()
    return choice != "0"


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
    Matches any term in the list.
    - Single words get word boundaries to reduce false positives.
    - Phrases match as-is.
    """
    cleaned = [t.strip() for t in terms if t.strip()]
    if not cleaned:
        return re.compile(r"(?!x)x")  # match nothing

    escaped = [re.escape(t) for t in cleaned]
    patterns = []
    for t, e in zip(cleaned, escaped):
        if " " in t:
            patterns.append(e)
        else:
            patterns.append(rf"\b{e}\b")
    return re.compile("|".join(patterns), flags=re.IGNORECASE)

def mask_word(word: str, mask_char: str = "*") -> str:
    return mask_char * len(word)

def censor_list_terms(text: str, censor_re: re.Pattern, mask_char: str) -> Tuple[str, List[str]]:
    """
    Censor all list terms immediately.
    Returns (censored_text, list_matches_original_forms)
    """
    matches = [m.group(0) for m in censor_re.finditer(text)]

    censored = censor_re.sub(lambda m: mask_word(m.group(0), mask_char), text)
    return censored, matches


# ============================================================
# AI per-word censoring (approximation)
# ============================================================
def split_tokens(text: str) -> List[str]:
    # keeps punctuation/spaces as separate tokens
    return re.findall(r"\w+|\W+", text)

def is_word_token(tok: str) -> bool:
    # For "word-like" tokens; keeps it simple
    return tok.strip().isalnum()

def predict_safe(model: Detoxify, text: str) -> Dict[str, float]:
    # Detoxify sometimes emits warnings; we just use predict output
    return model.predict(text)

def censor_toxic_words_ai(
    text: str,
    model: Detoxify,
    ai_threshold: float,
    mask_char: str,
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Censor ONLY words whose per-word toxicity >= ai_threshold.
    Returns (censored_text, ai_censored_words_with_scores)
    """
    tokens = split_tokens(text)
    ai_hits: List[Tuple[str, float]] = []
    out: List[str] = []

    for tok in tokens:
        if is_word_token(tok):
            # If it is already masked (e.g., ******), don't AI-score it.
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

def main():
    p = argparse.ArgumentParser(description="DE-SENSITIZER CLI (list censor + AI toxic-word censor)")
    p.add_argument("--terms", required=True, help="Path to terms.txt")
    p.add_argument("--mask", default="*", help="Mask character (default: *)")
    p.add_argument("--model", default="original", choices=["original", "unbiased", "multilingual"], help="Detoxify model")
    p.add_argument("--ai-threshold", type=float, default=0.60, help="Per-word toxicity threshold for AI censoring")
    p.add_argument("--show-lebron", action="store_true", help="Show LeBron ASCII vibe on startup")
    args = p.parse_args()

    show_big_banner()
    if args.show_lebron:
        show_lebron_ascii()

    type_out(f"{C.CYAN}Initializing Philippine modern government system...{C.RESET}", delay=0.02)

    if not press_to_begin():
        type_out(f"{C.YELLOW}Censorship cancelled. Bye!{C.RESET}", delay=0.02)
        return

    # Load list terms first (fast)
    terms = load_terms(args.terms)
    censor_re = build_censor_regex(terms)

    # Loading bar BEFORE model load
    loading_bar(seconds=1.6, width=32, label="Preparing yung model")

    # Load model
    type_out(f"{C.CYAN}Loading Detoxify model: {args.model} ...{C.RESET}", delay=0.02)
    model = Detoxify(args.model)
    type_out(f"{C.BRIGHT_GREEN}tapos na yung loading bossing, simula na tayo.{C.RESET}\n", delay=0.02)

    # Show config
    print(f"{C.BOLD}AI toxic-word trigger threshold:{C.RESET} {C.BRIGHT_YELLOW}{args.ai_threshold}{C.RESET}")
    print(f"{C.BOLD}List terms loaded:{C.RESET} {C.BRIGHT_YELLOW}{len(terms)}{C.RESET}")

    while True:
        text = input(f"{C.BOLD}Type text to desensitize:{C.RESET} ").rstrip("\n")
        if not text.strip():
            continue

        # Always censor list terms first
        list_censored_text, list_matches = censor_list_terms(text, censor_re, args.mask)

        # AI censors only toxic words (that are NOT already masked)
        final_censored, ai_hits = censor_toxic_words_ai(
            list_censored_text,
            model=model,
            ai_threshold=args.ai_threshold,
            mask_char=args.mask
        )

        # Full-message scores (for ratings)
        msg_scores = predict_safe(model, text)
        msg_tox = float(msg_scores.get("toxicity", 0.0))

        # Output
        print(f"\n{C.BOLD}=== RESULT ==={C.RESET}")
        print(f"{C.BOLD}AI toxic-word threshold:{C.RESET} {C.BRIGHT_YELLOW}{args.ai_threshold}{C.RESET}")

        # Decision label
        if list_matches:
            decision = f"{C.BRIGHT_RED}{C.BOLD}BLOCKED BY LIST{C.RESET}"
        elif msg_tox >= 0.80:
            decision = f"{C.BRIGHT_RED}BLOCK{C.RESET}"
        elif msg_tox >= 0.50:
            decision = f"{C.BRIGHT_YELLOW}REVIEW{C.RESET}"
        else:
            decision = f"{C.BRIGHT_GREEN}ALLOW{C.RESET}"


        if list_matches:
            print(f"{C.BOLD}Decision:{C.RESET} {decision}")
        else:
            print(f"{C.BOLD}Decision:{C.RESET} {decision}  {C.DIM}(message toxicity={msg_tox:.3f}){C.RESET}")

        print(f"{C.BOLD}Censored text:{C.RESET} {final_censored}")

        # Why censored
        print(f"\n{C.BOLD}CENSOR REPORT{C.RESET}")
        if not list_matches and not ai_hits:
            print(f"{C.DIM}- No words censored.{C.RESET}")
        else:
            if list_matches:
                # show unique while keeping order
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
                # unique tokens (best-effort) while keeping first score
                seen = set()
                uniq_ai: List[Tuple[str, float]] = []
                for w, s in ai_hits:
                    lw = w.lower()
                    if lw not in seen:
                        seen.add(lw)
                        uniq_ai.append((w, s))
                for w, s in uniq_ai:
                    print(f"- {C.BRIGHT_MAGENTA}'{w}'{C.RESET} censored by {C.BRIGHT_MAGENTA}[AI]{C.RESET} (word-tox={s:.3f})")

        # Show top scores (optional but useful)
        print(f"\n{C.BOLD}MESSAGE SCORES{C.RESET}")
        for k, v in format_scores(msg_scores)[:6]:
            # print top 6
            print(f"  - {k:14s} {v:.3f}")

        if not after_each_output_prompt():
            type_out(f"{C.YELLOW}Stopping. byebye.{C.RESET}", delay=0.02)
            break


if __name__ == "__main__":
    main()
