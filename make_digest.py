#!/usr/bin/env python3
"""
Turn the two HTML reports into a short plain-text digest (Markdown).

Used by the GitHub Action to open a daily issue, which GitHub then emails.
Reads docs/index.html + docs/stocks.html, writes digest.md and prints a title
line to $GITHUB_OUTPUT so the workflow can use it as the issue title.
"""

import os
import re
import sys
from datetime import date
from html import unescape


def text_blocks(path: str, stop_at: str) -> list[str]:
    """Extract the plain-language paragraphs/cards, ignoring the raw tables."""
    try:
        html = open(path).read()
    except FileNotFoundError:
        return []
    head = html.split(stop_at)[0]
    # each summary lives in a <p> or a card <div>; grab their inner text
    chunks = re.findall(r"<(?:p|div)[^>]*>(.*?)</(?:p|div)>", head, flags=re.S)
    out = []
    for c in chunks:
        c = re.sub(r"<br\s*/?>", " ", c)
        c = unescape(re.sub(r"<[^>]+>", "", c))
        c = re.sub(r"\s+", " ", c).strip()
        if len(c) > 40 and "not financial advice" not in c.lower():
            out.append(c)
    return out


def main():
    idx = text_blocks("docs/index.html", "Numbers behind it")
    stk = text_blocks("docs/stocks.html", "same thing as a table")

    header = ("> **Backtested and rejected**: +22% vs +849% for buy-and-hold over "
              "10 years. Observations only — not trade ideas.\n")
    lines = [header, f"## Market ({date.today():%a %d %b %Y})", ""]
    try:                       # paper portfolio section goes first when present
        pf = open("docs/portfolio_digest.txt").read().strip()
        if pf:
            lines = [header, pf, "", f"## Market ({date.today():%a %d %b %Y})", ""]
    except FileNotFoundError:
        pass
    lines += [f"- {b}" for b in idx] or ["- (index report unavailable)"]
    lines += ["", "## Stock candidates", ""]
    lines += [f"- {b}" for b in stk] or ["- Nothing today."]
    lines += [
        "",
        "---",
        "Full pages: "
        "[market](https://alpcanaras.github.io/swings-canner/) · "
        "[stocks](https://alpcanaras.github.io/swings-canner/stocks.html) · "
        "[paper portfolio](https://alpcanaras.github.io/swings-canner/portfolio.html)",
        "",
        "_Statistical tilts, not instructions. Skip anything flagged for earnings. "
        "Not financial advice._",
    ]
    body = "\n".join(lines)
    open("digest.md", "w").write(body)

    # headline for the issue title
    bias = "no signal"
    for b in idx:
        m = re.search(r"NDX.*?(lean toward buying dips|lean toward selling rallies|no trade)", b)
        if m:
            bias = "NDX " + m.group(1)
            break
    n = sum(1 for b in stk if re.match(r"^[A-Z\-]{1,6} — ", b))
    title = f"Scan {date.today():%Y-%m-%d} — {bias}, {n} stock candidate(s)"

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"title={title}\n")
    print(title, file=sys.stderr)


if __name__ == "__main__":
    main()
