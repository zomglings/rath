# rath
This is my harness. There are many like it, but this one is mine.

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

```
rath test                 # run all integration tests
rath test -n <name>       # run specific tests (repeatable; --name)
rath test --list          # list available tests
rath <command> -h         # help for any (sub)command
```

Integration tests are standalone scripts in `src/integration/` (compiled to
`dist/integration/`); each exits 0 on success. They call live APIs and require
`OPENAI_API_KEY`.
