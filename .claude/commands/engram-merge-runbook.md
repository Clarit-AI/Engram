---
description: Fetch and display the Engram Upstream Merge Runbook from Linear, inline in the session.
---
<objective>
Fetch the canonical **Engram Upstream Merge Runbook** from Linear and display
its body inline in the session, so the merge/upstream-sync steps are available
without leaving the terminal.
</objective>

<process>
1. Call the Linear MCP `get_document` tool with:
   - `id`: `ee75fc6b-1389-4824-8b7f-35485b8d8d43`
2. Display the returned document body inline, verbatim, in the session.
3. At the end, print the canonical Linear URL:

   `https://linear.app/khaentertainment/document/engram-upstream-merge-runbook-a2451e78eecc`

If the fetch fails (auth, network, or not found), report the error and the
canonical URL so the user can open it manually — do not fabricate runbook
content.
</process>
