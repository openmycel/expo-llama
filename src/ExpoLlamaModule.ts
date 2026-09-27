import { NativeModule, requireNativeModule } from "expo";

import type {
  ChatMessage,
  ExpoLlamaEvents,
  GenerateOptions,
  GenerateResult,
  LoadOptions,
  ModelInfo,
  SourceCheck,
} from "./ExpoLlama.types";

declare class ExpoLlamaModule extends NativeModule<ExpoLlamaEvents> {
  loadModel(path: string, options: LoadOptions): Promise<ModelInfo>;
  generate(messages: ChatMessage[], options: GenerateOptions): Promise<GenerateResult>;
  stop(): void;
  unload(): Promise<void>;
  isLoaded(): Promise<boolean>;
  sha256File(path: string): Promise<string>;
  excludeFromBackup(path: string): void;
  checkSource(url: string): Promise<SourceCheck>;
}

export default requireNativeModule<ExpoLlamaModule>("ExpoLlama");
