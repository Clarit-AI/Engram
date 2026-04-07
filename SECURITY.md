# Security Policy

## Supported Versions

Engram is pre-1.0. Security fixes are applied to the latest release only.

| Version | Supported |
| ------- | --------- |
| v0.1.x (latest) | Yes |
| Unreleased / main | Best-effort |
| Older tags | No |

When a new minor version ships (e.g., v0.2.0 after an upstream sync), the previous version leaves support. If you are running Engram in production or on edge hardware, pin to the latest tagged release and update when new versions land.

## Scope

Engram extends [SGLang](https://github.com/sgl-project/sglang) with persistent Mamba/SSM state management. The attack surface specific to Engram includes:

- **State snapshot persistence.** Engram saves and restores full model hidden state to disk. A malicious or tampered snapshot file could affect model behavior on restore.
- **Snapshot storage and access.** Saved states may contain information derived from prior conversations. Unauthorized access to snapshot files is a data exposure risk.
- **API extensions.** Engram adds HTTP endpoints for snapshot save, load, delete, and restore. These endpoints inherit SGLang's server configuration but introduce new input paths.
- **Agent tool framework.** The Engram agent tools interact with the snapshot system programmatically. Unexpected input through tool calls could trigger unintended state operations.

Issues in upstream SGLang that do not involve Engram's additions should be reported to the [SGLang project](https://github.com/sgl-project/sglang/security) directly.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Use GitHub's private vulnerability reporting:

1. Go to [Engram Security Advisories](https://github.com/Clarit-AI/Engram/security/advisories).
2. Click **"Report a vulnerability."**
3. Describe the issue, including steps to reproduce, affected versions, and potential impact.

You can also email **security@clarit.ai** if you prefer.

### What to expect

- **Acknowledgement** within 48 hours confirming we received the report.
- **Triage update** within 7 days with severity assessment and expected timeline.
- **Fix or mitigation** shipped in a patch release. We will coordinate disclosure timing with the reporter.

If the vulnerability affects upstream SGLang, we will coordinate with the SGLang maintainers before any public disclosure.

### What qualifies

- Remote code execution through Engram's API extensions or snapshot system
- Unauthorized access to saved model state or snapshot files
- Denial of service against the snapshot persistence layer
- Data leakage through cross-session state references
- Authentication or authorization bypasses in Engram's endpoints

### What does not qualify

- Issues in upstream SGLang with no Engram-specific component
- Vulnerabilities requiring local filesystem access on a machine where the attacker already has shell access
- Theoretical attacks with no demonstrated proof of concept

## Security Practices

Engram follows these practices today:

- Fork additions are additive. Engram does not modify upstream SGLang's security-relevant code paths; it adds new fields, methods, and conditional blocks alongside them.
- The state validation framework (63 tests) checks snapshot integrity on restore.
- CI runs on every PR. Upstream-only GPU workflows are gated to prevent false passes on the fork.

## Acknowledgements

We credit all security reporters in the release notes for the version containing the fix, unless the reporter requests anonymity.
