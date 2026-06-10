from __future__ import annotations

import ast
from pathlib import Path


DISALLOWED_PREFIXES = (
    "jupyter_workbench.adapters",
    "jupyter_workbench.core",
    "jupyter_client",
    "nbformat",
)

FORBIDDEN_MOUSE_CT_PREFIX = "mouse_ct"



PROCESSING_MODULES = {
    "microct_analysis.processing.calibration",
    "microct_analysis.processing.dicom",
    "microct_analysis.processing.io",
    "microct_analysis.processing.morphology",
    # "microct_analysis.processing.orientation",  # removed: PCA orientation retired
    "microct_analysis.processing.preprocess",
    "microct_analysis.processing.profiles",
    "microct_analysis.processing.resample",
    "microct_analysis.processing.sanity",
    "microct_analysis.processing.segmentation",
    "microct_analysis.processing.surface",
    "microct_analysis.processing.threshold",
    "microct_analysis.processing.types",
    "microct_analysis.processing.watershed",
}

SCIENTIFIC_DEPENDENCIES = ("pydicom", "nibabel", "scipy", "skimage", "SimpleITK")

ALLOWED_CORE_MODULES = {
    "microct_analysis.__init__",
    "microct_analysis.domain.__init__",
    "microct_analysis.domain.artifact_contracts",
    "microct_analysis.domain.confidence",
    "microct_analysis.notebook_tasks.__init__",
    "microct_analysis.notebook_tasks.cleanup",
    "microct_analysis.workflows.__init__",
    "microct_analysis.workflows.explain",
    "microct_analysis.workflows.feedback",
    "microct_analysis.workflows.loading",
    "microct_analysis.workflows.review",
    "microct_analysis.workflows.schema",
}


def _iter_python_modules(src_root: Path):
    for path in sorted(src_root.rglob("*.py")):
        yield path, ".".join(path.relative_to(src_root.parent).with_suffix("").parts)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return imports


def test_module_scope_imports_stay_on_public_interfaces(microct_src: Path) -> None:
    discovered_modules = set()
    violations: dict[str, list[str]] = {}

    for path, module_name in _iter_python_modules(microct_src):
        discovered_modules.add(module_name)
        bad = [name for name in _imports(path) if name.startswith(DISALLOWED_PREFIXES)]
        if bad:
            violations[module_name] = bad

    assert ALLOWED_CORE_MODULES <= discovered_modules
    assert violations == {}


def test_package_has_no_mouse_ct_imports(microct_src: Path) -> None:
    violations: dict[str, list[str]] = {}

    for path, module_name in _iter_python_modules(microct_src):
        bad = [name for name in _imports(path) if name == FORBIDDEN_MOUSE_CT_PREFIX or name.startswith(f"{FORBIDDEN_MOUSE_CT_PREFIX}.")]
        if bad:
            violations[module_name] = bad

    assert violations == {}


def test_processing_modules_can_be_imported() -> None:
    for module_name in sorted(PROCESSING_MODULES):
        __import__(module_name)


def test_key_scientific_dependencies_are_available() -> None:
    for module_name in SCIENTIFIC_DEPENDENCIES:
        __import__(module_name)
