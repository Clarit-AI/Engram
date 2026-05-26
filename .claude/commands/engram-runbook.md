---
description: Fetch and display the Engram Feature & Issue Runbook from Linear, inline in the session.
---
<objective>
Fetch the canonical **Engram Feature & Issue Runbook** from Linear and display
its body inline in the session, so the lifecycle steps are available without
leaving the terminal.
</objective>

<process>
1. Call the Linear MCP `get_document` tool with:
   - `id`: `05ca8e01-1f9a-4a68-9010-4be54f511ce8`
2. Display the returned document body inline, verbatim, in the session.
3. At the end, print the canonical Linear URL:

   `https://linear.app/khaentertainment/document/engram-feature-and-issue-runbook-d8916813a279`

If the fetch fails (auth, network, or not found), report the error and the
canonical URL so the user can open it manually — do not fabricate runbook
content.
</process>
