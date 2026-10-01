import ExpoModulesCore
import UIKit

public class ExpoLlamaModule: Module {
  private let engine = LlamaEngine()
  // Set by cancelSha256File, read by the running hash before each chunk. One hash at a time.
  private var hashCancelled = false

  public func definition() -> ModuleDefinition {
    Name("ExpoLlama")

    Events("onToken", "onMemoryWarning")

    // iOS warns the app in the foreground before it kills it; the app unloads the model.
    OnCreate {
      NotificationCenter.default.addObserver(
        forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main
      ) { [weak self] _ in
        self?.sendEvent("onMemoryWarning", [:])
      }
    }

    AsyncFunction("loadModel") { (path: String, options: LoadOptions, promise: Promise) in
      self.engine.queue.async {
        do {
          promise.resolve(try self.engine.load(path: filePath(path), options: options))
        } catch {
          promise.reject(error)
        }
      }
    }

    AsyncFunction("generate") { (messages: [ChatMessage], options: GenerateOptions, promise: Promise) in
      self.engine.queue.async {
        do {
          let result = try self.engine.generate(messages: messages, options: options) { text in
            self.sendEvent("onToken", ["text": text])
          }
          promise.resolve(result)
        } catch {
          promise.reject(error)
        }
      }
    }

    // Not queued: it has to reach a running `generate`.
    Function("stop") {
      self.engine.requestStop()
    }

    AsyncFunction("unload") { (promise: Promise) in
      self.engine.queue.async {
        self.engine.unload()
        promise.resolve()
      }
    }

    AsyncFunction("isLoaded") { (promise: Promise) in
      self.engine.queue.async {
        promise.resolve(self.engine.isLoaded)
      }
    }

    AsyncFunction("sha256File") { (path: String, promise: Promise) in
      self.hashCancelled = false
      DispatchQueue.global(qos: .utility).async {
        do {
          promise.resolve(
            try ModelFiles.sha256(path: filePath(path), isCancelled: { self.hashCancelled }))
        } catch {
          promise.reject(error)
        }
      }
    }

    // The running sha256File rejects with HashCancelledException within one chunk.
    Function("cancelSha256File") {
      self.hashCancelled = true
    }

    Function("excludeFromBackup") { (path: String) throws in
      try ModelFiles.excludeFromBackup(path: filePath(path))
    }

    AsyncFunction("checkSource") { (url: URL, promise: Promise) in
      ModelFiles.checkSource(url: url) { promise.resolve($0) }
    }

    OnDestroy {
      NotificationCenter.default.removeObserver(self)
      self.engine.requestStop()
      self.engine.queue.sync { self.engine.unload() }
    }
  }
}

// expo-file-system hands out file:// URIs; llama.cpp wants a plain path.
private func filePath(_ uri: String) -> String {
  if uri.hasPrefix("file://"), let url = URL(string: uri) {
    return url.path
  }
  return uri
}

struct LoadOptions: Record {
  @Field var contextSize: Int = 4096
  // -1 offloads every layer to the GPU (Metal).
  @Field var gpuLayers: Int = -1
}

struct ChatMessage: Record {
  @Field var role: String = "user"
  @Field var content: String = ""
}

// Defaults follow llama.cpp's own sampling defaults.
struct GenerateOptions: Record {
  @Field var maxTokens: Int = 1024
  @Field var temperature: Double = 0.8
  @Field var topK: Int = 40
  @Field var topP: Double = 0.95
  @Field var minP: Double = 0.05
  @Field var presencePenalty: Double = 0
  @Field var seed: Int?
  // GBNF with a "root" rule; the output can only be what it allows.
  @Field var grammar: String?
}

final class ModelNotFoundException: GenericException<String> {
  override var reason: String { "No model file at \(param)" }
}

final class ModelLoadException: GenericException<String> {
  override var reason: String { "llama.cpp could not load the model at \(param)" }
}

final class ContextException: GenericException<Int> {
  override var reason: String { "Could not create a context of \(param) tokens: not enough memory?" }
}

final class HashCancelledException: Exception {
  override var reason: String { "The hash was cancelled." }
}

final class ModelNotLoadedException: Exception {
  override var reason: String { "No model is loaded. Call loadModel first." }
}

final class ChatTemplateException: Exception {
  override var reason: String { "The model has no chat template llama.cpp can apply" }
}

final class GrammarException: Exception {
  override var reason: String { "llama.cpp could not parse the grammar (GBNF, needs a root rule)" }
}

final class TokenizeException: Exception {
  override var reason: String { "Could not tokenize the prompt" }
}

final class PromptTooLongException: GenericException<(Int, Int)> {
  override var reason: String { "The prompt is \(param.0) tokens, the context holds \(param.1)" }
}

final class DecodeException: GenericException<Int32> {
  override var reason: String { "llama_decode failed with status \(param)" }
}
