"""
generate_uml.py

Parses Python source files in the project using the AST module,
extracts classes, methods, attributes, and their relationships,
and writes yUML-compatible text that can be pasted into https://yuml.me

Usage:
    python generate_uml.py
    # Then copy the contents of uml_output.txt into https://yuml.me/diagram/scruffy/class/draw

Supported yUML notation:
    [Class]                     — simple class box
    [Class|field1;field2|m1()]  — class with fields and methods
    [Class]^-[Other]            — inheritance (Other inherits Class)
    [Class]->[Other]            — association
    [Class]++-1>[Other]         — composition (1-to-many)
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
# Directories to scan (relative to project root)
SOURCE_DIRS = ["models", "data", "utils"]
# Files to scan individually (relative to project root)
EXTRA_FILES = ["train.py", "scripts/inference.py"]

# Maximum number of fields / methods to show per class (keeps diagram readable)
MAX_FIELDS = 6
MAX_METHODS = 8

# yUML output file
OUTPUT_FILE = "uml_output.txt"

# Directories to ignore (venv, cache, etc.)
SKIP_DIRS = {"venv", ".git", ".claude", "__pycache__", "tests"}


# ---------------------------------------------------------------------------
# AST Helpers
# ---------------------------------------------------------------------------

def get_name(node) -> str:
    """Return a string name for an AST node (Name, Attribute, Constant, etc.)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{get_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Subscript):
        return f"{get_name(node.value)}[...]"
    if isinstance(node, ast.Call):
        return f"{get_name(node.func)}()"
    if isinstance(node, ast.BinOp):
        return "expr"
    return ""


def is_torch_module(node) -> bool:
    """Check if an AST node refers to a nn.Module subclass."""
    name = get_name(node)
    return any(
        name.endswith(suffix)
        for suffix in ("nn.Module", "Module", "nn.ModuleDict", "nn.ModuleList")
    )


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

class ClassInfo:
    """Stores parsed information about a Python class."""

    def __init__(self, name: str, module: str, bases=None):
        self.name = name
        self.module = module  # e.g. "models.vhoip"
        self.bases = bases or []  # list of base class names (strings)
        self.fields = []  # instance attributes set in __init__
        self.methods = []  # method names
        self.compositions = set()  # class names instantiated/assigned as attributes
        self.associations = set()  # other referenced classes

    def full_name(self) -> str:
        return f"{self.module}.{self.name}" if self.module else self.name

    def short_name(self) -> str:
        return self.name


def extract_classes_from_file(path: Path, module_prefix: str) -> list[ClassInfo]:
    """Parse a single .py file and return a list of ClassInfo objects."""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  WARN: Could not read {path}: {e}")
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  WARN: Syntax error in {path}: {e}")
        return []

    classes: list[ClassInfo] = []

    # First pass: collect all top-level class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [get_name(base) for base in node.bases]
            info = ClassInfo(name=node.name, module=module_prefix, bases=bases)

            # Walk class body
            for item in node.body:
                # Methods
                if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                    info.methods.append(item.name)

                    # If __init__, extract field assignments (self.x = ...)
                    if item.name == "__init__":
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                                        field_name = target.attr
                                        # Try to infer type from RHS
                                        rhs_name = get_name(stmt.value)
                                        if rhs_name and rhs_name != "":
                                            info.fields.append(f"{field_name}: {rhs_name}")
                                            # Check if RHS is a class instantiation -> composition
                                            if isinstance(stmt.value, ast.Call):
                                                call_name = get_name(stmt.value.func)
                                                if call_name and call_name[0].isupper():
                                                    info.compositions.add(call_name)
                                        else:
                                            info.fields.append(field_name)

                            elif isinstance(stmt, ast.AnnAssign):
                                if isinstance(stmt.target, ast.Attribute) and isinstance(stmt.target.value, ast.Name) and stmt.target.value.id == "self":
                                    field_name = stmt.target.attr
                                    ann = get_name(stmt.annotation) if stmt.annotation else ""
                                    info.fields.append(f"{field_name}: {ann}" if ann else field_name)

                # Class-level attributes (often module instantiations)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            rhs = get_name(item.value)
                            if rhs and rhs[0].isupper():
                                info.compositions.add(rhs)

            classes.append(info)

    return classes


def discover_all_classes() -> list[ClassInfo]:
    """Walk the project source directories and extract all classes."""
    all_classes: list[ClassInfo] = []

    # Scan package directories
    for src_dir in SOURCE_DIRS:
        dir_path = PROJECT_ROOT / src_dir
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            # Skip __pycache__ and similar
            if any(part in SKIP_DIRS for part in py_file.parts):
                continue
            rel = py_file.relative_to(PROJECT_ROOT)
            # Module name: e.g. "models.losses"
            module = str(rel.with_suffix("")).replace(os.sep, ".").replace("/", ".")
            classes = extract_classes_from_file(py_file, module)
            all_classes.extend(classes)
            print(f"  Scanned {rel}: {len(classes)} class(es)")

    # Scan extra top-level files
    for filename in EXTRA_FILES:
        file_path = PROJECT_ROOT / filename
        if file_path.exists():
            module = Path(filename).stem
            classes = extract_classes_from_file(file_path, module)
            all_classes.extend(classes)
            print(f"  Scanned {filename}: {len(classes)} class(es)")

    return all_classes


# ---------------------------------------------------------------------------
# yUML Builder
# ---------------------------------------------------------------------------

def build_yuml(classes: list[ClassInfo]) -> str:
    """Convert extracted class information into yUML text."""

    # Build lookup tables
    class_names = {c.short_name() for c in classes}
    class_by_name = {c.short_name(): c for c in classes}

    lines: list[str] = []

    # 1. Class boxes with members
    for c in classes:
        fields_str = ";".join(c.fields[:MAX_FIELDS])
        methods_str = ";".join([m + "()" for m in c.methods[:MAX_METHODS]])

        parts = []
        if fields_str:
            parts.append(fields_str)
        if methods_str:
            parts.append(methods_str)

        if parts:
            body = "|" + "|".join(parts)
            line = f"[{c.short_name()}{body}]"
        else:
            line = f"[{c.short_name()}]"

        lines.append(line)

    # 2. Inheritance relationships (^-)
    for c in classes:
        for base in c.bases:
            # Resolve base name (could be module.Class or just Class)
            base_short = base.split(".")[-1]
            if base_short in class_names and base_short != c.short_name():
                lines.append(f"[{base_short}]^-[{c.short_name()}]")

    # 3. Composition relationships (++-1>)
    for c in classes:
        for comp in c.compositions:
            comp_short = comp.split(".")[-1]
            if comp_short in class_names and comp_short != c.short_name():
                lines.append(f"[{c.short_name()}]++-1>[{comp_short}]")

    # 4. Association relationships (->)
    # Heuristic: if a method mentions another class name (outside compositions)
    for c in classes:
        for method in c.methods:
            # Skip dunder methods for associations
            if method.startswith("__") and method.endswith("__"):
                continue
            # Check if method name contains another class name (weak heuristic)
            for other in class_names:
                if other != c.short_name() and other.lower() in method.lower():
                    # Only add if not already composition
                    if other not in {comp.split(".")[-1] for comp in c.compositions}:
                        lines.append(f"[{c.short_name()}]->[{other}]")

    # Deduplicate relationships while preserving order
    seen_rels = set()
    unique_lines: list[str] = []
    for line in lines:
        if line.startswith("[") and ("^" in line or "->" in line or "++" in line):
            if line not in seen_rels:
                seen_rels.add(line)
                unique_lines.append(line)
        else:
            unique_lines.append(line)

    # yUML expects lines separated by commas for the URL format,
    # but for the diagram/draw box we can use newlines.
    # We will output both a "pasted" newline version and a comma URL version.
    return "\n".join(unique_lines)


def build_yuml_url(classes: list[ClassInfo]) -> str:
    """Build a yUML.me URL with comma-separated directives."""
    text = build_yuml(classes)
    # Replace newlines with commas for the URL format
    url_body = ",".join(text.strip().splitlines())
    # yUML.me scruffy class diagram URL
    return f"https://yuml.me/diagram/scruffy/class/draw/{url_body}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Discovering Python classes...")
    classes = discover_all_classes()
    print(f"\nTotal classes found: {len(classes)}")

    if not classes:
        print("No classes found. Exiting.")
        sys.exit(1)

    print("\nGenerating yUML text...")
    yuml_text = build_yuml(classes)
    yuml_url = build_yuml_url(classes)

    output_path = PROJECT_ROOT / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== yUML Text (paste into https://yuml.me/diagram/scruffy/class/draw) ===\n")
        f.write("\n")
        f.write("One per line format:\n")
        f.write(yuml_text)
        f.write("\n\n")
        f.write("=== Direct URL (may be too long for browser) ===\n")
        f.write(yuml_url)
        f.write("\n")

    print(f"\nOutput written to: {output_path}")
    print(f"\nInstructions:")
    print("  1. Open https://yuml.me/diagram/scruffy/class/draw")
    print("  2. Copy the content between the class boxes and paste it into the text box")
    print("  3. Click 'Draw' to generate the diagram")
    print(f"\nURL (try if short enough): {yuml_url[:200]}...")


if __name__ == "__main__":
    main()
