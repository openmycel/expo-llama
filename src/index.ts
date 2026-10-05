import ExpoLlama from './ExpoLlamaModule'
import type {
	ChatMessage,
	EmbedderInfo,
	EmbedderOptions,
	Embeddings,
	GenerateOptions,
	GenerateResult,
	LoadOptions,
	ModelInfo,
	SourceCheck,
} from './ExpoLlama.types'

export * from './ExpoLlama.types'

/** Loads a GGUF file (path or file:// URI). Replaces the model loaded before. */
export function loadModel(
	path: string,
	options: LoadOptions = {},
): Promise<ModelInfo> {
	return ExpoLlama.loadModel(path, options)
}

/**
 * Answers the conversation in `messages` with the model's chat template.
 * `onText` receives the answer piece by piece as it is generated.
 */
export async function generate(
	messages: ChatMessage[],
	options: GenerateOptions = {},
	onText?: (text: string) => void,
): Promise<GenerateResult> {
	const subscription = onText
		? ExpoLlama.addListener('onToken', (event) => onText(event.text))
		: null
	try {
		return await ExpoLlama.generate(messages, options)
	} finally {
		subscription?.remove()
	}
}

/** Ends a running `generate`; it resolves with what was written so far. */
export function stop(): void {
	ExpoLlama.stop()
}

export function unload(): Promise<void> {
	return ExpoLlama.unload()
}

export function isLoaded(): Promise<boolean> {
	return ExpoLlama.isLoaded()
}

/**
 * Loads an embedding model (GGUF) next to the chat model: both stay in memory, each with its
 * own weights. Replaces the embedding model loaded before.
 */
export function loadEmbedder(
	path: string,
	options: EmbedderOptions = {},
): Promise<EmbedderInfo> {
	return ExpoLlama.loadEmbedder(path, options)
}

/**
 * Vectors of `texts` from the embedding model, one per text, each of unit length. A model
 * that wants a prefix or an instruction (e5's "query: ", Qwen3-Embedding's "Instruct: …")
 * gets it in the text.
 */
export function embed(texts: string[]): Promise<Embeddings> {
	return ExpoLlama.embed(texts)
}

export function unloadEmbedder(): Promise<void> {
	return ExpoLlama.unloadEmbedder()
}

export function isEmbedderLoaded(): Promise<boolean> {
	return ExpoLlama.isEmbedderLoaded()
}

/** SHA-256 of a file (path or file:// URI) as lowercase hex, computed natively in chunks. */
export function sha256File(path: string): Promise<string> {
	return ExpoLlama.sha256File(path)
}

/**
 * Stops the running sha256File: it rejects with `HashCancelledException` within one 4 MB
 * chunk. One hash runs at a time.
 */
export function cancelSha256File(): void {
	ExpoLlama.cancelSha256File()
}

/**
 * Called when iOS warns that memory is low (UIApplication.didReceiveMemoryWarningNotification):
 * the moment to `unload()` before the app is killed. Returns the unsubscribe.
 */
export function onMemoryWarning(listener: () => void): () => void {
	const sub = ExpoLlama.addListener('onMemoryWarning', listener)
	return () => sub.remove()
}

/** Keeps a downloaded model out of iCloud and device backups. */
export function excludeFromBackup(path: string): void {
	ExpoLlama.excludeFromBackup(path)
}

/** One HEAD request to a download URL: can the phone reach it, and if not, why. */
export function checkSource(url: string): Promise<SourceCheck> {
	return ExpoLlama.checkSource(url)
}
