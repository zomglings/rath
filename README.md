# rath
This is my harness. There are many like it, but this one is mine.

rath is a library for creating specialized agent loops. rath does not define
an agent loop itself: each specialized agent is its own program with its own
loop, built from the primitives defined here — API providers and the pieces
needed to process and execute tool calls. The first primitive is the
`openai-native` provider.

## openai-native provider

`@zomglings/rath` registers a custom [pi-ai](https://github.com/badlogic/pi-mono) API
provider, `openai-native`, that adds OpenAI's native (server-side, "hosted")
tools to pi-ai without forking it. It is built on the official `openai` SDK
(Responses API).

Supported hosted tools: web search (on by default), file search, code
interpreter, and image generation (opt-in via stream options). The full
enumeration of hosted tools lives in OpenAI's built-in tools documentation
(https://platform.openai.com/docs/guides/tools); our support is not meant to
be comprehensive — e.g. remote MCP is not supported.

- Citation annotations (`url_citation`, `file_citation`,
  `container_file_citation`) are captured as structured `citations` on text
  blocks.
- Raw hosted tool call output items (`web_search_call`,
  `code_interpreter_call`, ...) ride along as `hostedToolCall` blocks and are
  replayed verbatim to the API on later turns.
- Contexts survive JSON serialize/deserialize with citations intact.

```ts
import { stream } from "@earendil-works/pi-ai";
import {
  getCitations,
  getHostedToolCalls,
  openaiNativeModel,
  registerOpenAINative,
} from "@zomglings/rath";

registerOpenAINative();

const model = openaiNativeModel("gpt-5-mini");
const context = {
  messages: [
    {
      role: "user" as const,
      content: "What is the latest Node.js LTS? Cite sources.",
      timestamp: Date.now(),
    },
  ],
};
const message = await stream(model, context, { reasoningEffort: "low" }).result();
for (const block of message.content) {
  if (block.type === "text") {
    console.log(block.text, getCitations(block));
  }
}
console.log(getHostedToolCalls(message));
```

Contexts containing `hostedToolCall` blocks or citations are understood by the
provider that produced them. Handing such a context to a different model
(another provider, or another model of the same provider) flattens the
extended blocks to plain text via `flattenHostedContent` so the new model
keeps the search history; same-model replay stays byte-identical to preserve
the prompt cache.

## openrouter-native provider

`openrouter-native` is the same idea for OpenRouter's server-side web search:
a provider wrapping OpenRouter's `openrouter:web_search` server tool on the
Chat Completions API, capturing its `url_citation` annotations as the same
structured `citations`. pi-ai routes OpenRouter through the stock
`openai-completions` api, which drops those annotations; this provider
preserves them. Register with `registerOpenRouterNative()` and use
`openrouterNativeModel("openai/gpt-4o")`. It reuses the shared `hosted-tools`
machinery, so citations, trailers, and flatten-on-handoff work identically.

`openrouterNativeModel` validates the id against the **live** OpenRouter
catalogue when it has been primed (`ensureCatalogue()` fetches OpenRouter's
keyless `/api/v1/models`, cached in the SQLite config store with a freshness
window),
building the model from the live pricing/context metadata. This means ids
newer than pi-ai's bundled registry (e.g. `anthropic/claude-fable-5` before a
pi-ai bump) work, and unknown ids are rejected against the current list. When
the catalogue has not been primed (used as a library without `ensureCatalogue`,
or offline), it falls back to pi-ai's bundled `openrouter` registry.

## CLI

The `rath` CLI exists for people developing or using rath to:

1. Test that the rath implementation is up-to-date with the upstream LLM APIs
   (integration tests).
2. Help agents (e.g. Pi, Claude Code, Codex) build specialized agents using
   rath: the CLI gives an agent a scriptable way to explore and verify the
   harness while it works.

```
rath run                  # generic agent loop, interactive (pi-tui)
rath run -p <prompt>      # auto-submit a prompt; exit when its turn settles
rath test                 # run all integration tests
rath test -n <name>       # run specific tests (repeatable; --name)
rath test --list          # list available tests
rath <command> -h         # help for any (sub)command
```

### rath run

`rath run` starts a generic agent loop with nothing implicit: no skill
discovery, no context-file walking (AGENTS.md is never read). The model sees
exactly what the flags specify; the provider API key is the only input taken
from the environment. The one convenience default is tools: `--tools` enables
all client-side tools when omitted (pass `--tools none` to disable them, or a
list to choose), since rath development inside `rath run` wants them on hand.

- One frontend: a pi-tui interface (a TTY is required) with differential
  rendering, an editor input, selector overlays for the session commands,
  Ctrl+C interrupting the current turn instead of killing the session, and a
  statusline below the editor — model, a colored context-window gauge built
  from the last turn's token usage (cache reads/writes, fresh input, output),
  cwd, the git branch colored by working-tree state (red merge/rebase, yellow
  dirty, green clean), and the time the last turn finished.
- `-p <prompt>` auto-submits the prompt as the session's first message and
  ends the session when its turn settles (exit code 1 if the turn errored).
  Typing a prompt of your own during the run takes the session over and keeps
  it open. The model can end any session with `end_session`, and `/exit
  [message]` / `end_session({message})` print the parting message to stdout
  after the TUI releases the terminal — the last, clean, capturable line of
  the run.
- Models are explicit: `-m <provider>/<model-id>`. Without `-m`, the pinned
  default model (`/config default-model`) is used, falling back to the
  built-in `openai-native/gpt-5.5`. Any registered pi-ai provider works. At
  startup rath primes a live OpenRouter model catalogue (the keyless
  `/api/v1/models`), cached in the config store, so `/lsmodels` and
  openrouter-native model resolution reflect OpenRouter's current list rather
  than pi-ai's bundled snapshot. (openai-native and the stock providers use
  pi-ai's bundled registry, which carries the pricing/context metadata they
  need; OpenAI's `/v1/models` is unfiltered and metadata-less, so it is not a
  usable live source.)
- Every startup setting is also settable in-session (both frontends), so a
  session never has to be restarted to change configuration: `/config` shows
  the configuration (`/config default-model [spec|none]` pins or clears the
  persisted default model), `/sys [text]` shows or sets the system prompt,
  `/model [spec]` shows or switches the model, `/lsmodels [filter]` lists
  models, `/reasoning [level]` shows or sets the reasoning level
  (openai-native clamps it to the model's supported levels),
  `/websearch [on|off]` toggles hosted web search, `/tools [names|none]`
  shows or sets client-side tools, `/save [path]` writes the context now and
  saves there on exit, `/go`/`/slow` (or `/mode`) switch interaction mode,
  `/exit [message]` quits (printing the message after the TUI closes).
  Changes take effect on the next turn. Bare `/model` and `/reasoning` open
  selector overlays.
- Two interaction modes (`--mode go|slow`, default `go`): **go** runs at full
  speed, tools execute immediately. **slow** gates every tool call behind a
  per-call confirmation (also the mitigation for prompt-injection driving
  tools while web search is on) and pages long output through `$PAGER`,
  suspending the TUI for the pager's duration.
- Any registered pi-ai provider works, plus rath's own hosted-tool providers:
  `openai-native/<model>` and `openrouter-native/<model>` (OpenRouter's
  server-side web search with citations; see below).
- Hosted web search is on by default with openai-native (`--no-web-search`
  disables it). After each reply, citations are rendered into a `Sources:`
  text block appended to the assistant message, marked
  `renderedCitations: true`: it persists in saved contexts and flattens for
  free when the context is handed to a provider that does not understand
  citations, and it is stripped before replay to openai-native, which
  reconstructs the real annotations itself.
- `--tools` enables client-side tools (the full set:
  `read,bash,edit,write,grep,find,ls,request_human_edit,barbarian_review,configure,list_models,save_context,end_session`).
  Omitting `--tools` enables all of them; `--tools none` disables them. The
  first seven come from `@earendil-works/pi-coding-agent` and run with your
  privileges in the current directory; the rest are rath's own tools, which
  give the model the same controls over the session that you have through the
  slash commands — the agent operates the harness as a peer, not a passenger.
- `barbarian_review` spawns the Barbarian Reviewer (see `rath barbarian`
  below) from inside a session: the model can ask for an adversarial review
  of a repo's changes and gets the findings report back as the tool result,
  optionally choosing the barbarian's model and reasoning level.
- `request_human_edit` is rath's human-in-the-loop tool: it opens a file in
  your editor (`$VISUAL`/`$EDITOR`, falling back to the first of `code`, `vim`,
  `emacs`, `nano` on PATH; GUI editors like `code`/`cursor` get `--wait`
  appended so the call blocks) and waits for you to save and quit, then returns
  the final contents and a unified diff of your changes. The agent can seed the
  file with a draft via `content`, name a `path`, or let it use a temp file
  (whose path is returned either way). The TUI suspends while the editor runs.
- `configure` lets the model inspect or change its own session settings (model,
  reasoning, web search, mode, active tools, system prompt) and pin the
  persisted default model (`defaultModel`) for future sessions; calling it with
  no fields just reads the configuration. `list_models` enumerates the model
  catalog (the tool form of `/lsmodels`). `save_context` writes the session
  JSON to a path (the tool form of `/save`). `end_session` ends the session
  (the tool form of `/exit`). All are ordinary tool calls, so in slow mode they
  are gated behind the per-call confirmation — the model proposes, you approve
  — and in go mode they apply immediately.
- `--save <path>` writes the context as JSON on exit; `--load <path>` resumes
  from one.

### rath barbarian

`rath barbarian` runs the Barbarian Reviewer: a non-interactive subagent
(`src/agents/barbarian.ts`) that adversarially reviews the changes from a
source commit-ish to a target commit-ish in a git repository and writes a
findings report. It is not a linter — it hunts defects (correctness,
regressions, broken contracts, security exposure, bad tests, incomplete
changes) and is expected to prove findings by reproduction where possible,
staging disposable `git worktree`s under a temp artifact root.

- Defaults: `--source` is `main` (falling back to `master`); `--target` is
  the current repository state — staged, unstaged, and untracked changes
  captured as a synthetic commit in a disposable worktree, so the review
  covers work in progress without ever touching your tree.
- `--repo` points at any path inside the target repository (default: cwd).
- `--model` and `--reasoning` choose the barbarian's model (default: the
  pinned default model) and effort (default: `high`).
- `--output` names the findings file (relative to the repo root); by default
  it lands in the artifact root. `--instructions` appends extra reviewer
  instructions to the prompt.
- The findings report prints to stdout; progress and the artifact/findings
  paths go to stderr. Hosted web search is disabled for the barbarian (an
  unattended agent has no business following injectable web content).
- The same agent is available inside `rath run` as the `barbarian_review`
  tool.

## Integration tests

### Contract

Each integration test is a standalone script: it is executed with `node`,
receives no arguments, and signals its verdict purely through its exit code —
0 for pass, non-zero (conventionally 1) for fail. Anything it writes to
stdout/stderr is shown as-is; on failure the assertion error and stack trace
land on stderr.

### Discovery and execution

Tests live in `src/integration/`, one file per test, and compile with the
normal build (`npm run build`) to `dist/integration/`. `rath test` discovers
every `*.js` file in `dist/integration/` (resolved relative to the installed
CLI, so the published package can run its own tests); the test's name is its
filename without the extension. Tests run sequentially, each in a child
`node` process with inherited stdio and environment, and the runner reports
per-test PASS/FAIL plus a summary, exiting 1 if any test failed. To add a
test, drop a script in `src/integration/` that exits 0 on success and
rebuild — no registration step.

### Requirements and conventions

Most tests call the live OpenAI API: they require `OPENAI_API_KEY` in
the environment and fail fast with a clear error when it is missing. They
cost real money (fractions of a cent in tokens, plus per-use fees for hosted
tools such as web search and code interpreter containers), which is why they
are not part of the pre-commit checks. (Several tests are the exception:
`request-human-edit`, `config-preferences`, `configure-tool`, `session-tools`,
`slow-mode-gate`, `statusline`, and `barbarian-git` call no API and need no
key — they exercise the CLI tools and config store directly; `catalogue` needs network but no key, hitting
OpenRouter's keyless `/api/v1/models`, and skips cleanly when offline.)
`RATH_TEST_MODEL` overrides the model used by the
API tests (default: `gpt-5.5`; the OpenRouter test uses `openai/gpt-5.5`).
Tests log a per-run token cost on success, and
assert on request payloads via the provider's `onPayload` hook when they need
to prove what was actually sent to the API.

### Current tests

- `openai-native-web-search` — the issue #5 acceptance spike. Three turns
  through pi-ai's `stream()`: a hosted web search with structured citations,
  lossless replay of raw `web_search_call` items, and a JSON
  serialize/deserialize round-trip of the full context.
- `openai-native-code-interpreter` — opt-in hosted tool. Computes fib(100)
  with `codeInterpreter: true`, checks the `code_interpreter_call` item is
  captured and the answer exact, then replays it losslessly after a JSON
  round-trip.
- `openai-native-agent-loop` — interop with pi's stock agent loop
  (`@earendil-works/pi-agent-core`). Hosted tools disabled; a client-side
  tool is executed by the loop and only `function` tools appear in request
  payloads.
- `openai-native-client-tool` — client-side tool parsing through plain
  `stream()`: ToolCall block shape (structured arguments, `callId|itemId`
  id, `stopReason "toolUse"`), then `function_call`/`function_call_output`
  replay with matching `call_id`.
- `request-human-edit` — the request_human_edit tool, driven by a fake editor
  (no API, no key). Covers temp-file vs given-path round-trips (consistent
  return shape), the no-change case, and editor resolution ($VISUAL/$EDITOR
  precedence, `--wait` injection for GUI editors, no-editor error).
- `config-preferences` — the SQLite config store (no API, no key): config-dir
  resolution, default-model set/update/clear round-trip, automatic schema
  migration, re-open idempotency, and forward-compatibility.
- `catalogue` — the live model catalogue (network, no key): fetches and caches
  OpenRouter's `/api/v1/models`, validates openrouter-native against the live
  list (builds a model with per-million costs; rejects unknown ids), and reuses
  the cache within the freshness window. Skips cleanly when offline.
- `configure-tool` — the configure tool (no API, no key): every field applied
  to agent state and flags, rebuilding the tool set including configure
  itself, per-field error reporting, the empty-call no-op, and pinning/clearing
  the persisted default model.
- `session-tools` — the session-operating tools (no API, no key): list_models
  enumerates and filters the catalog, save_context writes the session JSON and
  sets the save-on-exit path, and end_session requests exit and terminates.
- `statusline` — the TUI statusline (no API, no key): context-bar
  apportionment (proportional, floored for non-zero categories, clamped),
  token/timestamp formatting, full-line rendering with and without usage, and
  `gitInfo` against throwaway repositories in each working-tree state (unborn
  branch, clean, dirty, detached HEAD, not-a-repo).
- `barbarian-git` — the Barbarian Reviewer's git plumbing (no API, no key):
  repo-root resolution, the main→master source fallback, change detection,
  and the synthetic target commit (staged + unstaged + untracked captured in
  a disposable worktree, with the user's tree left untouched and the diff
  covering exactly the working-tree changes).

Not yet covered: `file_search` (needs a vector-store fixture) and
`image_generation` (cost).
