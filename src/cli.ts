#!/usr/bin/env node
import { runSubcommands, type Command } from "./command.js";
import { testCommand } from "./commands/test.js";

const root: Command = {
  name: "rath",
  summary: "An agent harness",
  description: "rath is an agent harness built on pi-ai with custom API providers.",
  subcommands: [testCommand],
  run: runSubcommands,
};

Promise.resolve(root.run("", process.argv.slice(2))).then(
  (code) => process.exit(code),
  (error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  },
);
