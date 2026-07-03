# microct-analysis

This repo is a Meridian prompt package for micro-CT study design and analysis. The current branch carries the converged agent-package design: a Study Design agent turns researcher intent into a protocol, an Analyst executes that protocol with explain-before-execute provenance, and a Code Reviewer checks scientific correctness, code behavior, and provenance completeness.

## Source of truth

- Edit source package files in `agents/`, `skills/`, `bootstrap/`, and docs.
- Do not edit `.mars/`, `.claude/`, `.codex/`, or `.opencode/` as source; those are generated or target surfaces.
- Root `skills/` holds micro-CT package skills. Dependency skills come from imported packages.

## Meridian command targeting

Meridian sessions may inherit a project root that is not the shell CWD. Always target the intended package or scenario workspace explicitly with `meridian -C <project-root> ...` when running Mars or launching agents.

Examples from this repo:

```bash
# Validate this source package, regardless of inherited Meridian state.
meridian -C /home/jimyao/gitrepos/prompts/microct-analysis mars validate

# Sync the local package into the scenario workspace.
meridian -C /home/jimyao/gitrepos/prompts/microct-arena mars sync

# If the scenario workspace does not already depend on this checkout, add it then sync.
meridian -C /home/jimyao/gitrepos/prompts/microct-arena mars add ../microct-analysis
meridian -C /home/jimyao/gitrepos/prompts/microct-arena mars sync

# Confirm the workspace sees the micro-CT agents from this package.
meridian -C /home/jimyao/gitrepos/prompts/microct-arena mars agents | grep microct
```

For relative paths, run from this repo and prefer:

```bash
meridian -C "$PWD" mars validate
meridian -C ../microct-arena mars sync
```

Launch scenario agents from the workspace, not from the package checkout, so they see the scenario files, datasets, and synced package targets:

```bash
meridian -C ../microct-arena spawn -a microct-study-design
meridian -C ../microct-arena spawn -a microct-analyst
meridian -C ../microct-arena spawn -a microct-code-reviewer
```

Use `mars sync --force` only when intentionally discarding local target modifications in the scenario workspace. Never hand-edit generated target files to make a launch work; fix source here, then sync the target project.

## Agent model

Agents are thin wrappers around cognitive mode, permissions, and loaded skills.

- `microct-study-design`: deliberative protocol design with the researcher.
- `microct-analyst`: procedural execution from an accepted protocol.
- `microct-code-reviewer`: adversarial review of code, measurements, and provenance.

Do not reintroduce stage-specialist agents for segmentation, landmarking, measurement, or cleanup unless arena evidence shows the cognitive split earns its handoff cost.

## Skill model

Skills hold reusable method and domain knowledge:

- Always-relevant package skills: `study-design`, `analysis-core`, `provenance`, `environment-setup`.
- Domain skills: `bone-morphometry`, `vascular-analysis`, `materials-analysis`.

Keep skill bodies short and route examples or detailed domain references to `resources/`.

## Runtime code status

The historical Python package under `src/microct_analysis/` is preserved for now because it is test-covered and contains useful processing and measurement knowledge. The target architecture is agent-generated analysis code with transparent provenance; convert runtime helpers into reference material deliberately, module by module, rather than deleting test-covered code blindly.
