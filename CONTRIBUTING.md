# Contributing

Thanks for taking the time to improve this HNSW implementation. The checklist below shows how we expect pull requests to be prepared so reviewers can focus on the proposed behavior changes instead of repo hygiene.

## 1. Set up your environment

- Fork the repository and clone your fork locally.
- Install the dependencies once via `npm install`.
- Run `npm test` to ensure the working tree passes before you start editing.

## 2. Create a focused branch

- Branch names should describe the change, for example `fix-level-selection` or `docs/pr-process`.
- Keep each branch focused on a single fix or feature so it is easy to review and revert if needed.

## 3. Develop and validate your change

- Keep code changes and documentation updates in the same commit when they belong together so context is not lost.
- Add or update tests under `tests/` that capture the bug fix or feature so regressions are caught automatically. The `tests/HNSW.test.ts` suite shows how to build deterministic indices for verification.
- Before committing, run the full set of quality gates:
  - `npm test` – runs the Jest harness
  - `npm run lint` – checks the TypeScript sources with TSLint
  - `npm run build` – ensures the TypeScript compiler can emit the distributable files

## 4. Commit with context

- Use present-tense, descriptive commit messages, e.g. `Add efSearch option to searchKNN`.
- Include any relevant benchmarking notes or data sources in the commit description if your change impacts performance or quality measurements.

## 5. Open the pull request

- Push your branch to your fork and open a PR against `main`.
- Fill in the PR template: summarize *what* changed and *why*, and list the exact commands you ran for testing.
- Link to any related issues (bug reports, feature requests, documentation gaps) so reviewers can see the full context.

## 6. Iterate with reviewers

- Apply review feedback in additional commits so conversations stay traceable.
- Re-run `npm test`, `npm run lint`, and `npm run build` after each revision that touches code.
- When the PR is approved, squash or rebase as requested by the maintainer, then merge.

## Releasing to npm (maintainers only)

After merging a PR to `main`, follow these steps to publish a new version:

### Choose the version bump

Use [semantic versioning](https://semver.org/):
- **Patch** (1.0.x): Bug fixes, no behavior changes
- **Minor** (1.x.0): New features, algorithm improvements, backward compatible
- **Major** (x.0.0): Breaking API changes

### Publish workflow

```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Run the version bump (runs lint, format, and creates a git tag)
npm version patch   # or minor, or major

# This automatically runs:
# - preversion: npm run lint
# - version: npm run format && git add -A src
# - postversion: git push && git push --tags

# Publish to npm (runs prepublishOnly: npm test && npm run lint)
npm publish

# If this is your first publish or you need to log in:
npm login
npm publish
```

The version scripts are already configured in `package.json` to handle the workflow, so `npm version` will automatically lint, format, commit, tag, and push.

Following this workflow keeps the project healthy and makes it clear how each contribution was validated before landing.
