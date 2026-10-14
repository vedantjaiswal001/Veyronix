# VEYRONIX — Safety & Security Model (spec §33/§53)

**The AI can PROPOSE; it can never arbitrarily EXECUTE.** This boundary is
structural, not advisory, and is enforced by the remediation layer
(`veyronix/remediation/`).

## Controls

1. **Allowlist (no arbitrary execution).** Actions are a fixed `ActionType`
   enum. There is no code path anywhere that executes an LLM-supplied string,
   shell command, or SQL. An unknown action string fails Pydantic validation
   before it can become a request.
2. **Parameter validation.** Each action's parameters are validated (service
   must be a real service; `replicas ∈ [1,20]`; `size_gb ∈ [1,4096]`; rollback
   needs a deployment or service). Invalid or malformed params are rejected at
   request construction *and* re-validated at execution (defense in depth).
3. **Risk tiers + gating.**
   - LOW → auto-eligible **only** when the global kill-switch
     (`VEYRONIX_ALLOW_AUTO_REMEDIATION`) is on; otherwise requires approval.
   - MEDIUM → requires human approval (role `approver`/`admin`).
   - HIGH → requires elevated (`admin`) approval.
   - **BLOCKED → never executes** (e.g. `DELETE_DATABASE`), at submit, approve
     and execute stages.
4. **Authorization.** Approval and execution are gated by actor role. An
   `operator` cannot approve a MEDIUM action; a `guest` cannot execute anything.
5. **Approval workflow + audit trail.** Every action records who proposed it,
   who approved/rejected it, when, why, and its predicted-vs-actual outcome
   (`actions` + `action_outcomes` tables); every transition emits a structured
   audit log.
6. **Kill-switch default off.** Nothing auto-executes unless explicitly enabled.

## Verified by tests (`tests/security/`)

- an arbitrary command / SQL string is not a valid action (rejected)
- `DELETE_DATABASE` cannot be submitted-to-run, approved, or executed
- MEDIUM action cannot execute before approval; operator cannot self-approve it
- unauthorized role cannot execute; rejected actions never execute
- invalid/malformed parameters are rejected
- the agent tool surface is entirely read-only (no mutating tool exists)

The LLM investigator's recommendation is only ever an allowlisted `ActionType`
that then enters this workflow — it is a proposal, subject to every control above.
