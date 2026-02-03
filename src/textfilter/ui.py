import argparse

from .core import TextFilter, format_scores
from .ui import C, after_each_output_prompt, loading_bar, press_to_begin, show_banner, type_out


def main() -> None:
    p = argparse.ArgumentParser(description="Text Filtering CLI (term list + Detoxify)")
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
    type_out(f"{C.CYAN}Initializing...{C.RESET}")

    if not press_to_begin():
        type_out(f"{C.YELLOW}Operation cancelled.{C.RESET}")
        return

    loading_bar(seconds=1.2, width=32, label="Loading resources")

    type_out(f"{C.CYAN}Loading Detoxify model: {args.model}...{C.RESET}")
    tf = TextFilter(
        terms_path=args.terms,
        model_name=args.model,
        mask_char=args.mask,
        ai_threshold=args.ai_threshold,
    )
    type_out(f"{C.BRIGHT_GREEN}Ready.{C.RESET}\n")

    print(f"{C.BOLD}AI per-word threshold:{C.RESET} {C.BRIGHT_YELLOW}{args.ai_threshold}{C.RESET}")
    print(f"{C.BOLD}List terms loaded:{C.RESET} {C.BRIGHT_YELLOW}{len(tf.terms)}{C.RESET}")

    while True:
        text = input(f"\n{C.BOLD}Enter text:{C.RESET} ").rstrip("\n")
        if not text.strip():
            continue

        result = tf.process(text)

        decision = str(result["decision"])
        msg_tox = float(result["toxicity"])
        censored_text = str(result["censored_text"])
        list_matches = list(result["list_matches"])
        ai_hits = list(result["ai_hits"])
        scores = dict(result["scores"])

        # Pretty decision label
        if decision == "BLOCKED_LIST":
            decision_label = f"{C.BRIGHT_RED}{C.BOLD}BLOCKED (LIST){C.RESET}"
        elif decision == "BLOCK":
            decision_label = f"{C.BRIGHT_RED}BLOCK{C.RESET}"
        elif decision == "REVIEW":
            decision_label = f"{C.BRIGHT_YELLOW}REVIEW{C.RESET}"
        else:
            decision_label = f"{C.BRIGHT_GREEN}ALLOW{C.RESET}"

        print(f"\n{C.BOLD}=== RESULT ==={C.RESET}")
        if decision == "BLOCKED_LIST":
            print(f"{C.BOLD}Decision:{C.RESET} {decision_label}")
        else:
            print(f"{C.BOLD}Decision:{C.RESET} {decision_label}  {C.DIM}(toxicity={msg_tox:.3f}){C.RESET}")

        print(f"{C.BOLD}Censored text:{C.RESET} {censored_text}")

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
                uniq_ai = []
                for w, s in ai_hits:
                    lw = w.lower()
                    if lw not in seen:
                        seen.add(lw)
                        uniq_ai.append((w, s))
                for w, s in uniq_ai:
                    print(
                        f"- {C.BRIGHT_MAGENTA}'{w}'{C.RESET} censored by "
                        f"{C.BRIGHT_MAGENTA}[AI]{C.RESET} (word-tox={float(s):.3f})"
                    )

        print(f"\n{C.BOLD}MESSAGE SCORES{C.RESET}")
        for k, v in format_scores(scores)[:6]:
            print(f"  - {k:14s} {float(v):.3f}")

        if not after_each_output_prompt():
            type_out(f"{C.YELLOW}Stopping.{C.RESET}")
            break
