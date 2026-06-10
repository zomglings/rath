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

Contexts containing `hostedToolCall` blocks or citations are only understood by
the `openai-native` provider; flatten them before handing a context to a stock
pi-ai provider (not yet implemented).

## CLI

The `rath` CLI exists for people developing or using rath to:

1. Test that the rath implementation is up-to-date with the upstream LLM APIs
   (integration tests).
2. Help agents (e.g. Pi, Claude Code, Codex) build specialized agents using
   rath: the CLI gives an agent a scriptable way to explore and verify the
   harness while it works.

```
rath test                 # run all integration tests
rath test -n <name>       # run specific tests (repeatable; --name)
rath test --list          # list available tests
rath <command> -h         # help for any (sub)command
```

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
  (`@earendil-works/pi-agent-core`, a dev dependency). Hosted tools disabled;
  a client-side tool is executed by the loop and only `function` tools appear
  in request payloads.
- `openai-native-client-tool` — client-side tool parsing through plain
  `stream()`: ToolCall block shape (structured arguments, `callId|itemId`
  id, `stopReason "toolUse"`), then `function_call`/`function_call_output`
  replay with matching `call_id`.

Not yet covered: `file_search` (needs a vector-store fixture) and
`image_generation` (cost).
