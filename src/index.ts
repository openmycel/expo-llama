import ExpoLlama from "./ExpoLlamaModule";
import type {
  ChatMessage,
  GenerateOptions,
  GenerateResult,
  LoadOptions,
  ModelInfo,
  SourceCheck,
} from "./ExpoLlama.types";

export * from "./ExpoLlama.types";

/** Loads a GGUF file (path or file:// URI). Replaces the model loaded before. */
export function loadModel(path: string, options: LoadOptions = {}): Promise<ModelInfo> {
  return ExpoLlama.loadModel(path, options);
}

/**
 * Answers the conversation in `messages` with the model's chat template.
 * `onText` receives the answer piece by piece as it is generated.
 */
export async function generate(
  messages: ChatMessage[],
  options: GenerateOptions = {},
  onText?: (text: string) => void
): Promise<GenerateResult> {
  const subscription = onText
    ? ExpoLlama.addListener("onToken", (event) => onText(event.text))
    : null;
  try {
    return await ExpoLlama.generate(messages, options);
  } finally {
    subscription?.remove();
  }
}

/** Ends a running `generate`; it resolves with what was written so far. */
export function stop(): void {
  ExpoLlama.stop();
}

export function unload(): Promise<void> {
  return ExpoLlama.unload();
}

export function isLoaded(): Promise<boolean> {
  return ExpoLlama.isLoaded();
}

/** SHA-256 of a file (path or file:// URI) as lowercase hex, computed natively in chunks. */
export function sha256File(path: string): Promise<string> {
  return ExpoLlama.sha256File(path);
}

/** Keeps a downloaded model out of iCloud and device backups. */
export function excludeFromBackup(path: string): void {
  ExpoLlama.excludeFromBackup(path);
}

/** One HEAD request to a download URL: can the phone reach it, and if not, why. */
export function checkSource(url: string): Promise<SourceCheck> {
  return ExpoLlama.checkSource(url);
}
