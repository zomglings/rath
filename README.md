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
`openrouterNativeModel("openai/gpt-4o")` (any id from pi-ai's `openrouter`
registry). It reuses the shared `hosted-tools` machinery, so citations,
trailers, and flatten-on-handoff work identically.

## CLI

The `rath` CLI exists for people developing or using rath to:

1. Test that the rath implementation is up-to-date with the upstream LLM APIs
   (integration tests).
2. Help agents (e.g. Pi, Claude Code, Codex) build specialized agents using
   rath: the CLI gives an agent a scriptable way to explore and verify the
   harness while it works.

```
rath run                  # generic agent loop, interactive
rath run -p <prompt>      # one prompt, then exit
rath test                 # run all integration tests
rath test -n <name>       # run specific tests (repeatable; --name)
rath test --list          # list available tests
rath <command> -h         # help for any (sub)command
```

### rath run

`rath run` starts a generic agent loop with nothing implicit: no skill
discovery, no context-file walking (AGENTS.md is never read), no tools unless
explicitly enabled with `--tools`. The model sees exactly what the flags
specify; the provider API key is the only input taken from the environment.
rath development itself is meant to happen inside `rath run`.

- Two interactive frontends: a plain readline REPL (default; also handles
  `-p` one-shots) and a pi-tui interface (`-T`/`--tui`) with differential
  rendering, an editor input, selector overlays for the session commands,
  and Ctrl+C interrupting the current turn instead of killing the session.
- Models are explicit: `-m <provider>/<model-id>` (default
  `openai-native/gpt-5-mini`; any registered pi-ai provider works).
- Every startup setting is also settable in-session (both frontends), so a
  session never has to be restarted to change configuration: `/info` shows
  the configuration, `/sys [text]` shows or sets the system prompt,
  `/model [spec]` shows or switches the model, `/lsmodels [filter]` lists
  models, `/reasoning [level]` shows or sets the reasoning level
  (openai-native clamps it to the model's supported levels),
  `/websearch [on|off]` toggles hosted web search, `/tools [names|none]`
  shows or sets client-side tools, `/save [path]` writes the context now and
  saves there on exit, `/go`/`/slow` (or `/mode`) switch interaction mode,
  `/exit` quits. Changes take effect on the next turn. In the TUI, bare
  `/model` and `/reasoning` open selector overlays.
- Two interaction modes (`--mode go|slow`, default `go`): **go** runs at full
  speed, tools execute immediately. **slow** gates every tool call behind a
  per-call confirmation (also the mitigation for prompt-injection driving
  tools while web search is on) and pages long output — through `$PAGER` in
  the plain REPL, a scrollable overlay in the TUI.
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
- `--tools read,bash,edit,write,grep,find,ls` enables client-side tools from
  `@earendil-works/pi-coding-agent`. They run with your privileges in the
  current directory.
- `--save <path>` writes the context as JSON on exit; `--load <path>` resumes
  from one.

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

The current tests call the live OpenAI API: they require `OPENAI_API_KEY` in
the environment and fail fast with a clear error when it is missing. They
cost real money (fractions of a cent in tokens, plus per-use fees for hosted
tools such as web search and code interpreter containers), which is why they
are not part of the pre-commit checks. `RATH_TEST_MODEL` overrides the model
(default: `gpt-5-mini`). Tests log a per-run token cost on success, and
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

Not yet covered: `file_search` (needs a vector-store fixture) and
`image_generation` (cost).
