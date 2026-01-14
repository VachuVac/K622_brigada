#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

def unique_target_path(dst_dir: Path, name: str) -> Path:
    """Vrátí unikátní cestu v dst_dir – pokud soubor existuje, přidá ' (1)', ' (2)', ..."""
    base = Path(name).stem
    ext = Path(name).suffix
    candidate = dst_dir / f"{base}{ext}"
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{base} ({i}){ext}"
        i += 1
    return candidate

def iter_source_files(root: Path, include_las: bool) -> list[Path]:
    """Najde všechny .laz (a volitelně .las) mimo kořenový adresář (tj. jen v podsložkách)."""
    patterns = ["*.laz"] + (["*.las"] if include_las else [])
    files = []
    for pat in patterns:
        for p in root.rglob(pat):
            # přeskoč soubory přímo v rootu (už jsou „nahoře“)
            if p.parent.resolve() == root.resolve():
                continue
            if p.is_file():
                files.append(p)
    return files

def main():
    ap = argparse.ArgumentParser(
        description="Přesun/zkopírování všech LAZ (volitelně LAS) z podsložek do nadložky."
    )
    ap.add_argument("root", type=Path, help="Cesta k nadložce (kořenové složce).")
    ap.add_argument("--action", choices=["move", "copy"], default="move",
                    help="Co dělat se soubory: move (výchozí) nebo copy.")
    ap.add_argument("--include-las", action="store_true",
                    help="Zahrnout i .las (nejen .laz).")
    ap.add_argument("--no-dry-run", action="store_true",
                    help="Vykonat akce doopravdy (bez tohoto běží jen náhled).")
    ap.add_argument("--verbose", action="store_true", help="Ukecanější výstup.")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        ap.error(f"Nadložka neexistuje nebo není složka: {root}")

    to_process = iter_source_files(root, include_las=args.include_las)
    if not to_process:
        print("Nenalezeny žádné odpovídající soubory v podsložkách.")
        return

    print(f"Nalezeno souborů: {len(to_process)}")
    dry = not args.no_dry_run

    for src in to_process:
        dst = unique_target_path(root, src.name)
        rel_src = src.relative_to(root)
        rel_dst = dst.relative_to(root)
        if args.verbose or dry:
            print(f"{args.action.upper():<4}  {rel_src}  ->  {rel_dst}")
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.action == "move":
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))

    if dry:
        print("\nDRY-RUN hotov (nic se nepřesunulo). Přidejte --no-dry-run pro provedení.")
    else:
        print("\nHotovo.")

if __name__ == "__main__":
    main()
