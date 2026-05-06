# Contributing

Thanks for considering a contribution to MemPoison.

This repository is still a research repository, so the contribution standard is
slightly different from a production library:

- preserve experiment provenance
- prefer small, explicit changes
- keep benchmark claims aligned with the actual evidence

## Recommended Workflow

1. Create a focused branch.
2. Keep changes scoped to one purpose:
   - code cleanup
   - benchmark tooling
   - documentation
3. Run the smallest relevant validation before opening a PR.
4. In the PR description, explain:
   - what changed
   - why it changed
   - what you validated

## Repository Areas

- `src/mempoison/`
  Preferred place for reusable code.
- `scripts/`
  Good place for small utilities and repo tasks.

## Validation Expectations

Use the lightest validation that matches the change:

- docs-only change:
  - check links, paths, and commands
- package/code change:
  - import the affected module
  - run a small local smoke if possible

## Style Notes

- Python: keep code direct and explicit
- Docs: prefer concrete instructions over high-level slogans
- Claims: do not strengthen scientific claims without matching evidence

## Release Note

The source code is licensed under MIT. Treat full benchmark releases,
provenance bundles, and paper materials as separately reviewed artifacts unless
they are explicitly added to this public tree.
