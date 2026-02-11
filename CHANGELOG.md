# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- CI workflow for build, lint, test, and coverage.
- Persistence-focused test coverage for IndexedDB-backed index behavior.
- API reference and tuning guidance in README.

### Changed
- Lint gate now targets published library sources (`src/**`, excluding benchmark CLI code).
- README persistence example now loads from the same DB name it saved to.

### Fixed
- `HNSWWithDB.deleteIndex()` now awaits DB re-initialization.
- `HNSWWithDB` now surfaces initialization/load/delete errors instead of silently swallowing them.
