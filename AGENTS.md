# microct-analysis

This repo is a Meridian prompt package for micro-CT study design and analysis. The current branch carries the converged agent-package design: a Study Design agent turns researcher intent into a protocol, an Analyst executes that protocol with explain-before-execute provenance, and a Code Reviewer checks scientific correctness, code behavior, and provenance completeness.

## Source of truth

- Edit source package files in `agents/`, `skills/`, `bootstrap/`, and docs.
- Do not edit `.mars/`, `.claude/`, `.codex/`, or `.opencode/` as source; those are generated or target surfaces.
- Root `skills/` holds micro-CT package skills. Dependency skills come from imported packages.

## Meridian command targeting

Meridian sessions may inherit a project root that is not the shell CWD. Always target the intended package or arena explicitly with `meridian -C <project-root> ...` when running Mars or launching agents.

Examples from this repo:

```bash
# Validate this source package, regardless of inherited Meridian state.
meridian -C /home/jimyao/gitrepos/prompts/microct-analysis mars validate

# Sync the local package into the blind evaluation arena.
meridian -C /home/jimyao/gitrepos/prompts/microct-arena mars sync

# If the arena does not already depend on this checkout, add it then sync.
meridian -C /home/jimyao/gitrepos/prompts/microct-arena mars add ../microct-analysis
meridian -C /home/jimyao/gitrepos/prompts/microct-arena mars sync

# Confirm the arena sees the micro-CT agents from this package.
meridian -C /home/jimyao/gitrepos/prompts/microct-arena mars agents | grep microct
```

For relative paths, run from this repo and prefer:

```bash
meridian -C "$PWD" mars validate
meridian -C ../microct-arena mars sync
```

Launch evaluation agents from the arena, not from the package checkout, so they see the blind scenarios, datasets, and synced package targets:

```bash
meridian -C ../microct-arena spawn -a microct-study-design
meridian -C ../microct-arena spawn -a microct-analyst
meridian -C ../microct-arena spawn -a microct-code-reviewer
```

Use `mars sync --force` only when intentionally discarding local target modifications in the arena. Never hand-edit generated target files to make a launch work; fix source here, then sync the target project.

## Testing in microct-arena

Use `../microct-arena` as the blind evaluation harness for this package. The package checkout defines agents and skills; the arena checkout defines scenarios, datasets, runs, and hidden evaluation answers.

Arena setup:

```bash
# From this repo, sync the current package into the arena.
meridian -C ../microct-arena mars sync
meridian -C ../microct-arena mars agents | grep microct
```

Blind run workflow:

```bash
cd ../microct-arena
git checkout main
git checkout -b exp/<name>
git rm -rf evaluations/
git commit -m "strip answer key for blind run"
meridian -C "$PWD" spawn -a microct-study-design
```

Feed the visible scenario, not the answer key. For the first wedge, use:

```bash
scenarios/scenario-1-trabecular-morphometry.md
```

Run the workflow as the researcher:

1. Start with `microct-study-design`; judge whether it asks the right questions and designs the right protocol from the vague scenario.
2. Hand the accepted protocol to `microct-analyst`; judge whether it applies the readiness gate, probes the data/environment, explains before execution, and records provenance.
3. Hand protocol + analyst artifacts to `microct-code-reviewer`; judge whether it catches scientific, code, measurement, QC, and provenance issues.
4. Commit the run artifacts on the experiment branch.
5. Score outside the agent context by comparing the run against `main:evaluations/scenario-1/expected.yaml` and `main:evaluations/scenario-1/scoring.md`.

Keep the answer key blind. Do not expose `evaluations/` to the agent, do not paste expected answers into prompts, and do not run the agent from an arena branch where evaluation files are present. If using tmux, keep answer-key inspection in a separate human-only pane.

What the arena is testing first: study-design reasoning. Measurement accuracy matters, but the primary signal is whether a generic prompt package can infer the right analysis plan, compartments, medial/lateral comparison, subchondral VOI, QC evidence, and explanation/provenance requirements from an underspecified research question.

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
