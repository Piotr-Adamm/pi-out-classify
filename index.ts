import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerOutputHook } from "./src/hook.ts";
import { registerReadOverride } from "./src/read.ts";

export default function outClassify(pi: ExtensionAPI): void {
  registerReadOverride(pi);
  registerOutputHook(pi);
}
