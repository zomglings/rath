/**
 * PATH lookup, the way a shell resolves a bare command name. Used to detect
 * which optional external programs (pager, editor) are available so callers
 * can pick a working default instead of hard-coding one.
 */
import { existsSync, statSync } from "node:fs";
import { delimiter, isAbsolute, join } from "node:path";

/**
 * Return true if an executable named `cmd` exists. A name containing a path
 * separator (or an absolute path) is checked directly; a bare name is searched
 * across PATH entries. On Windows, PATHEXT extensions (.exe/.cmd/...) are tried
 * for names without their own extension. This only checks for an existing
 * file; it never runs the candidate.
 */
export function isOnPath(cmd: string): boolean {
  const exts =
    process.platform === "win32"
      ? (process.env.PATHEXT ?? ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean)
      : [""];
  const hasExt = exts.some((e) => e !== "" && cmd.toLowerCase().endsWith(e.toLowerCase()));
  const names = hasExt ? [cmd] : exts.map((e) => cmd + e);
  const isFile = (p: string): boolean => {
    try {
      return existsSync(p) && statSync(p).isFile();
    } catch {
      return false;
    }
  };
  if (cmd.includes("/") || cmd.includes("\\") || isAbsolute(cmd)) {
    return names.some(isFile);
  }
  const dirs = (process.env.PATH ?? "").split(delimiter).filter(Boolean);
  return dirs.some((dir) => names.some((n) => isFile(join(dir, n))));
}
