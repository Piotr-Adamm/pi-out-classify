import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerCodemodeGoal } from "./src/codemode.ts";
import { registerOutputHook } from "./src/hook.ts";
import { registerReadOverride } from "./src/read.ts";

export default function outClassify(pi: ExtensionAPI): void {
  registerReadOverride(pi);
  registerCodemodeGoal(pi);
  registerOutputHook(pi);
}
