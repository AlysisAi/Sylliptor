# Changelog

All notable changes to Sylliptor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.9.1] - 2026-06-08

### Added

- Native API protocols for OpenAI, Anthropic, and Gemini provider profiles.
- Provider-native and external web search backends, including Tavily when configured.
- Router model selection in setup and configuration so lightweight routing can use a cheaper model.

### Fixed

- Hardened provider diagnostics, Forge/runtime status reporting, asset display, and cross-platform test behavior.
- Improved configuration, profile, MCP, hook, and tool output handling for clearer public CLI behavior.

## [0.9.0b2] - 2026-05-09

### Fixed

- Restored the sandbox image workflow for GHCR publishing and runtime smoke validation.
- Fixed sandbox image smoke checks by avoiding nested container init handling.
- Updated release metadata so the next beta can publish as a new PyPI version.

### Changed

- Cleaned stale beta support wording and public issue template version examples.

## [0.9.0b1] - 2026-05-09

### Added

- First beta package published to PyPI as `sylliptor-agent-cli`.
- GitHub Actions release workflow with build, test, package smoke, and PyPI Trusted Publisher publishing.
- Public governance files, contribution templates, funding metadata, and polished docs navigation.

### Changed

- Public-facing repository documentation was trimmed and polished for beta launch.
- README and docs navigation now point visitors to the public Sylliptor website.

## [0.1.4] - 2026-04-06

### Added

- Mutable chat session workdirs for changing task context during a session.
- Live surface feedback, including a thinking spinner and streamed Markdown rerendering.

### Fixed

- Plan Mode entry flow, exact plan subcommand parsing, and natural-language workdir navigation.
- Review approval keyboard navigation.

## [0.1.3] - 2026-04-02

### Added

- Production-grade Skills validation flow and installed CLI smoke coverage.
- Skills lifecycle guidance for first-time authoring and installation flows.

### Fixed

- Invalid `/plan` guidance and blank MCP prompt server filter handling.

## [0.1.2] - 2026-03-31

### Added

- First Skills MVP with authoring, lifecycle, validation, and evaluation support.
- MCP foundations for stdio and streamable HTTP, including tools, roots, resources, and prompts.
- Tier 1 custom tools foundation and bundled model catalog provenance.

### Changed

- Chat Plan Mode, repo grounding, and approval flows were tightened for clearer read-only planning.
- Forge and swarm execution were hardened around task scope, verification, and planner handoff.

### Fixed

- MCP transport completion edge cases, protected follow-up synthesis, and atomic runtime artifact handling.

## [0.1.1] - 2026-03-18

### Added

- Core local agent CLI with chat and run modes, model provider API access, tool execution, streaming responses, image inputs, clipboard support, and slash commands.
- Setup, configuration, usage tracking, conversation compaction, history search, and workspace binding flows.
- Forge planning and execution workflows with swarm orchestration, worktrees, verification gates, review gates, conflict review, and feedback bundle export.
- Sandboxed shell execution, isolated server worker jobs, web fetch/search tools, subagents, and the first extensions foundation.

### Changed

- Rebranded the package and visible CLI surfaces from the initial Coder naming to Sylliptor and Forge.
- Hardened runtime safety across git operations, protected paths, shell policy, workspace scope, sandboxing, and verification evidence.

### Fixed

- Cross-platform verification, terminal UX, task routing, prompt handling, and recovery paths across managed execution flows.

## [0.1.0] - 2026-02-13

### Added

- Initial Python package scaffold for a local coding agent CLI.
- Baseline README, architecture notes, Apache-2.0 license, packaging metadata, and development configuration.
