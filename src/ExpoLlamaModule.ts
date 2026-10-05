import { NativeModule, requireNativeModule } from 'expo'

import type {
	ChatMessage,
	EmbedderInfo,
	EmbedderOptions,
	Embeddings,
	ExpoLlamaEvents,
	GenerateOptions,
	GenerateResult,
	LoadOptions,
	ModelInfo,
	SourceCheck,
} from './ExpoLlama.types'

declare class ExpoLlamaModule extends NativeModule<ExpoLlamaEvents> {
	loadModel(path: string, options: LoadOptions): Promise<ModelInfo>
	generate(
		messages: ChatMessage[],
		options: GenerateOptions,
	): Promise<GenerateResult>
	stop(): void
	unload(): Promise<void>
	isLoaded(): Promise<boolean>
	loadEmbedder(path: string, options: EmbedderOptions): Promise<EmbedderInfo>
	embed(texts: string[]): Promise<Embeddings>
	unloadEmbedder(): Promise<void>
	isEmbedderLoaded(): Promise<boolean>
	sha256File(path: string): Promise<string>
	cancelSha256File(): void
	addListener<K extends keyof ExpoLlamaEvents>(
		event: K,
		listener: ExpoLlamaEvents[K],
	): { remove(): void }
	excludeFromBackup(path: string): void
	checkSource(url: string): Promise<SourceCheck>
}

export default requireNativeModule<ExpoLlamaModule>('ExpoLlama')
