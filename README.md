# microct-analysis

Meridian prompt package for micro-CT study design and analysis.

The current architecture is a three-agent workflow:

1. `microct-study-design` turns a researcher's question into an executable protocol.
2. `microct-analyst` runs the accepted protocol with transparent generated code, QC evidence, and provenance.
3. `microct-code-reviewer` checks scientific validity, code behavior, measurements, and provenance against the protocol.

Reusable method and domain knowledge live in `skills/`:

- `study-design`
- `analysis-core`
- `provenance`
- `environment-setup`
- `bone-morphometry`
- `vascular-analysis`
- `materials-analysis`

The historical Python package under `src/microct_analysis/` is still present as test-covered runtime/reference material. The target prompt architecture keeps scientific decisions visible in generated analysis code rather than hiding them behind opaque domain APIs.

## Working on the package

Edit source files in `agents/` and `skills/`. Do not edit generated target directories such as `.mars/`, `.claude/`, `.codex/`, or `.opencode/` directly.

Useful checks:

```bash
git diff --check
meridian mars validate
```

Before publishing, run `meridian mars check` from a clean package export or a checkout without generated target/runtime directories.
