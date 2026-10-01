export type ChatRole = "system" | "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type LoadOptions = {
  /** Context window in tokens. Default 4096. */
  contextSize?: number;
  /** Layers offloaded to the GPU (Metal); -1 = all. Default -1. Ignored in the iOS Simulator: CPU only. */
  gpuLayers?: number;
};

export type ModelInfo = {
  description: string;
  sizeBytes: number;
  parameters: number;
  contextSize: number;
};

export type GenerateOptions = {
  /** Default 1024. */
  maxTokens?: number;
  /** 0 = greedy. Default 0.8. */
  temperature?: number;
  /** Default 40. */
  topK?: number;
  /** Default 0.95. */
  topP?: number;
  /** Default 0.05. */
  minP?: number;
  /** Penalizes tokens already used in the last 64; 0 = off. Default 0. */
  presencePenalty?: number;
  seed?: number;
  /**
   * GBNF grammar (llama.cpp's format) with a `root` rule: the answer can only be text the
   * grammar allows, e.g. one JSON shape. Rejects the call if the grammar does not parse.
   */
  grammar?: string;
};

export type GenerateResult = {
  text: string;
  promptTokens: number;
  tokens: number;
  /** True when `stop()` ended the generation. */
  stopped: boolean;
  promptMs: number;
  generationMs: number;
  tokensPerSecond: number;
};

export type TokenEvent = {
  text: string;
};

export type ExpoLlamaEvents = {
  onToken: (event: TokenEvent) => void;
  /** iOS is low on memory: unload the model before the app is killed. */
  onMemoryWarning: () => void;
};

export type SourceCheck = {
  ok: boolean;
  /**
   * Why the source failed: `offline` — the phone has no connection; `unreachable` — the host
   * cannot be reached (DNS, TLS, timeout, HTTP 403/451: what a block looks like);
   * `http` — the server answered with another error status.
   */
  reason?: "offline" | "unreachable" | "http";
  status?: number;
  /** URLError code, when there is one. */
  code?: number;
  message?: string;
};
