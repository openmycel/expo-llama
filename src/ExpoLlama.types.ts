export type ChatRole = 'system' | 'user' | 'assistant'

export type ChatMessage = {
	role: ChatRole
	content: string
}

export type LoadOptions = {
	/** Context window in tokens. Default 4096. */
	contextSize?: number
	/** Layers offloaded to the GPU (Metal); -1 = all. Default -1. Ignored in the iOS Simulator: CPU only. */
	gpuLayers?: number
}

export type EmbedderOptions = {
	/** Tokens one text may have; a longer text is cut. Default 512. */
	contextSize?: number
	/** Layers offloaded to the GPU (Metal); -1 = all. Default -1. CPU only in the Simulator. */
	gpuLayers?: number
	/**
	 * How the tokens' vectors become one: `mean`, `cls` or `last` — what the model's card
	 * says (Qwen3-Embedding: `last`, e5: `mean`). Default: what the model file says.
	 */
	pooling?: 'mean' | 'cls' | 'last'
	/**
	 * CPU threads it computes on; the chat model keeps the others. Default 2: a short text
	 * needs few, and the two models do not fight for the cores.
	 */
	threads?: number
}

export type EmbedderInfo = ModelInfo & {
	/** The length of each vector. */
	dimensions: number
}

export type Embeddings = {
	/** One vector per text, in order, each of unit length: a dot product is the cosine. */
	vectors: number[][]
	/** Tokens read, all texts together. */
	tokens: number
	ms: number
}

export type ModelInfo = {
	description: string
	sizeBytes: number
	parameters: number
	contextSize: number
}

export type GenerateOptions = {
	/** Default 1024. */
	maxTokens?: number
	/** 0 = greedy. Default 0.8. */
	temperature?: number
	/** Default 40. */
	topK?: number
	/** Default 0.95. */
	topP?: number
	/** Default 0.05. */
	minP?: number
	/** Penalizes tokens already used in the last 64; 0 = off. Default 0. */
	presencePenalty?: number
	seed?: number
	/**
	 * GBNF grammar (llama.cpp's format) with a `root` rule: the answer can only be text the
	 * grammar allows, e.g. one JSON shape. Rejects the call if the grammar does not parse.
	 */
	grammar?: string
	/**
	 * A context of its own on the loaded model, made on first use: the weights are shared,
	 * the KV cache is its own. Each session keeps what it last read, so a prompt that starts
	 * the same way as the one before (the same system prompt, a longer chat) is read only
	 * from where it differs. Use one per kind of prompt — "router" for a fixed instruction
	 * asked again and again, the default "" for the chat — so they do not overwrite each
	 * other's cache. Default "".
	 */
	session?: string
	/**
	 * The session's context size in tokens: its KV cache is allocated for all of them, so a
	 * session that only reads short prompts can take less memory than the chat. Asked with
	 * another size than the session has, the session is made again and its cache is lost.
	 * Default: the `contextSize` of `loadModel`.
	 */
	contextSize?: number
}

export type GenerateResult = {
	text: string
	promptTokens: number
	/** Tokens of the prompt taken from the session's cache, not read again. */
	cachedTokens: number
	tokens: number
	/** True when `stop()` ended the generation. */
	stopped: boolean
	promptMs: number
	generationMs: number
	tokensPerSecond: number
}

export type TokenEvent = {
	text: string
}

export type ExpoLlamaEvents = {
	onToken: (event: TokenEvent) => void
	/** iOS is low on memory: unload the model before the app is killed. */
	onMemoryWarning: () => void
}

export type SourceCheck = {
	ok: boolean
	/**
	 * Why the source failed: `offline` — the phone has no connection; `unreachable` — the host
	 * cannot be reached (DNS, TLS, timeout, HTTP 403/451: what a block looks like);
	 * `http` — the server answered with another error status.
	 */
	reason?: 'offline' | 'unreachable' | 'http'
	status?: number
	/** URLError code, when there is one. */
	code?: number
	message?: string
}
