# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Stamps the website's own scripts and stylesheets with a hash of their contents (site.js?v=1a2b3c4d),
in every page in docs/, so that a browser holding an older copy fetches the new one after an
update; GitHub Pages lets browsers keep a file for ten minutes. Run after changing the site:

    python3 bench/site_stamp.py
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
REF = re.compile(r'((?:src|href)=")([\w.-]+\.(?:js|css))(?:\?v=[0-9a-f]+)?(")')


def main():
    hashes = {}
    for page in sorted(DOCS.glob("*.html")):
        text = page.read_text()

        def stamp(m):
            name = m.group(2)
            f = DOCS / name
            if not f.exists():
                return m.group(0)
            if name not in hashes:
                hashes[name] = hashlib.sha256(f.read_bytes()).hexdigest()[:8]
            return f"{m.group(1)}{name}?v={hashes[name]}{m.group(3)}"

        new = REF.sub(stamp, text)
        if new != text:
            page.write_text(new)
            print(f"stamped {page.name}")
    for name, h in sorted(hashes.items()):
        print(f"  {name} {h}")


if __name__ == "__main__":
    main()
