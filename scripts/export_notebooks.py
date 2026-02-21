from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_notebook(notebook_path: Path, output_dir: Path) -> Path:
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    lines: list[str] = [
        f"# Auto-exported from {notebook_path.as_posix()}",
        "# Source notebook retained for exploratory and narrative context.",
        "",
    ]

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            lines.append(f"# %% [markdown] cell {index}")
            for markdown_line in source.splitlines():
                lines.append(f"# {markdown_line}" if markdown_line else "#")
            lines.append("")
            continue

        if cell_type == "code":
            lines.append(f"# %% code cell {index}")
            if source.strip():
                lines.extend(source.rstrip("\n").splitlines())
            lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{notebook_path.stem}.py"
    script_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return script_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all Jupyter notebooks in a folder to Python scripts.",
    )
    parser.add_argument("--input", default="notebooks", help="Notebook directory")
    parser.add_argument("--output", default="scripts", help="Export directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    notebooks = sorted(input_dir.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No notebooks found in: {input_dir}")

    for notebook in notebooks:
        exported = export_notebook(notebook, output_dir)
        print(f"Exported: {exported}")


if __name__ == "__main__":
    main()