import Foundation
import llama

// Level of the last llama.cpp log line: GGML_LOG_LEVEL_CONT lines (progress dots)
// continue it and are filtered the same way.
private var lastLogLevel = GGML_LOG_LEVEL_NONE

// A context of its own on the loaded model, by name: its KV cache keeps what it last read,
// so the next prompt that starts the same way is read only from where it differs.
private final class Session {
  let context: OpaquePointer
  /// The context size it was asked for; llama.cpp may round what it allocates.
  let size: Int
  /// The tokens in the KV cache of sequence 0, in order.
  var cached: [llama_token] = []
  init(_ context: OpaquePointer, size: Int) {
    self.context = context
    self.size = size
  }
}

// One model at a time, and a context per session on it ("" is the default one). Every call
// runs on `queue`, so load, generate and unload never overlap; only `requestStop` crosses
// threads, guarded by `stopLock`.
final class LlamaEngine {
  let queue = DispatchQueue(label: "org.openmycel.expo-llama", qos: .userInitiated)

  private var model: OpaquePointer?
  private var contextParams = llama_context_default_params()
  // A second model of its own, for vectors (an embedding model), and its context.
  private var embedder: OpaquePointer?
  private var embedContext: OpaquePointer?
  private var sessions: [String: Session] = [:]
  private let stopLock = NSLock()
  private var stopRequested = false

  private static let backend: Void = {
    // llama.cpp logs every graph node at load; keep warnings and errors only.
    llama_log_set({ level, text, _ in
      if level != GGML_LOG_LEVEL_CONT { lastLogLevel = level }
      guard let text, lastLogLevel.rawValue >= GGML_LOG_LEVEL_WARN.rawValue else { return }
      fputs(text, stderr)
    }, nil)
    llama_backend_init()
  }()

  init() {
    _ = LlamaEngine.backend
  }

  deinit {
    unload()
    unloadEmbedder()
  }

  var isLoaded: Bool { model != nil }

  func requestStop() {
    stopLock.lock()
    stopRequested = true
    stopLock.unlock()
  }

  private func takeStop() -> Bool {
    stopLock.lock()
    defer { stopLock.unlock() }
    return stopRequested
  }

  private func resetStop() {
    stopLock.lock()
    stopRequested = false
    stopLock.unlock()
  }

  func load(path: String, options: LoadOptions) throws -> [String: Any] {
    unload()
    guard FileManager.default.fileExists(atPath: path) else {
      throw ModelNotFoundException(path)
    }

    var modelParams = llama_model_default_params()
    #if targetEnvironment(simulator)
    // Metal in the iOS Simulator returns garbage for llama.cpp (only "!" tokens, b11146,
    // Xcode 27); the CPU gives correct output. On a device the GPU is used.
    modelParams.n_gpu_layers = 0
    #else
    modelParams.n_gpu_layers = Int32(options.gpuLayers)
    #endif
    guard let model = llama_model_load_from_file(path, modelParams) else {
      throw ModelLoadException(path)
    }

    var contextParams = llama_context_default_params()
    contextParams.n_ctx = UInt32(options.contextSize)
    // Leave two cores for the UI and the system.
    let threads = Int32(max(1, ProcessInfo.processInfo.activeProcessorCount - 2))
    contextParams.n_threads = threads
    contextParams.n_threads_batch = threads
    guard let context = llama_init_from_model(model, contextParams) else {
      llama_model_free(model)
      throw ContextException(options.contextSize)
    }

    self.model = model
    self.contextParams = contextParams
    sessions[""] = Session(context, size: options.contextSize)

    var desc = [CChar](repeating: 0, count: 256)
    llama_model_desc(model, &desc, desc.count)
    return [
      "description": String(cString: desc),
      "sizeBytes": Double(llama_model_size(model)),
      "parameters": Double(llama_model_n_params(model)),
      "contextSize": Int(llama_n_ctx(context)),
    ]
  }

  func unload() {
    for session in sessions.values { llama_free(session.context) }
    sessions = [:]
    if let model { llama_model_free(model) }
    model = nil
  }

  // An embedding model next to the chat model: its own weights and one context that pools
  // each text into one vector. The whole text goes in one batch (n_batch = n_ubatch =
  // contextSize): an encoder such as BERT reads it at once, not in pieces.
  func loadEmbedder(path: String, options: EmbedderOptions) throws -> [String: Any] {
    unloadEmbedder()
    guard FileManager.default.fileExists(atPath: path) else {
      throw ModelNotFoundException(path)
    }
    var modelParams = llama_model_default_params()
    #if targetEnvironment(simulator)
    modelParams.n_gpu_layers = 0
    #else
    modelParams.n_gpu_layers = Int32(options.gpuLayers)
    #endif
    guard let model = llama_model_load_from_file(path, modelParams) else {
      throw ModelLoadException(path)
    }
    var params = llama_context_default_params()
    params.embeddings = true
    params.n_ctx = UInt32(options.contextSize)
    params.n_batch = UInt32(options.contextSize)
    params.n_ubatch = UInt32(options.contextSize)
    params.pooling_type = options.pooling.flatMap { Self.pooling[$0] } ?? LLAMA_POOLING_TYPE_UNSPECIFIED
    // Few threads: a short text needs little, and the chat model keeps the rest.
    let threads = Int32(max(1, min(options.threads, ProcessInfo.processInfo.activeProcessorCount - 2)))
    params.n_threads = threads
    params.n_threads_batch = threads
    guard let context = llama_init_from_model(model, params) else {
      llama_model_free(model)
      throw ContextException(options.contextSize)
    }
    guard llama_pooling_type(context) != LLAMA_POOLING_TYPE_NONE else {
      llama_free(context)
      llama_model_free(model)
      throw PoolingException()
    }
    embedder = model
    embedContext = context
    var desc = [CChar](repeating: 0, count: 256)
    llama_model_desc(model, &desc, desc.count)
    return [
      "description": String(cString: desc),
      "sizeBytes": Double(llama_model_size(model)),
      "parameters": Double(llama_model_n_params(model)),
      "contextSize": Int(llama_n_ctx(context)),
      "dimensions": Int(llama_model_n_embd(model)),
    ]
  }

  private static let pooling: [String: llama_pooling_type] = [
    "mean": LLAMA_POOLING_TYPE_MEAN, "cls": LLAMA_POOLING_TYPE_CLS, "last": LLAMA_POOLING_TYPE_LAST,
  ]

  var isEmbedderLoaded: Bool { embedder != nil }

  func unloadEmbedder() {
    if let embedContext { llama_free(embedContext) }
    if let embedder { llama_model_free(embedder) }
    embedContext = nil
    embedder = nil
  }

  // One vector per text, each of unit length, so a dot product is the cosine. A text longer
  // than the context is cut to it.
  func embed(texts: [String]) throws -> [String: Any] {
    guard let model = embedder, let context = embedContext else {
      throw EmbedderNotLoadedException()
    }
    let started = DispatchTime.now()
    let vocab = llama_model_get_vocab(model)
    let size = Int(llama_n_ctx(context))
    let dimensions = Int(llama_model_n_embd(model))
    let encoder = llama_model_has_encoder(model) && !llama_model_has_decoder(model)
    var vectors: [[Double]] = []
    var tokenCount = 0
    for text in texts {
      // The model's own BOS and EOS: some pool on the last token, which must be EOS.
      let bytes = Int32(text.utf8.count)
      var tokens = [llama_token](repeating: 0, count: Int(bytes) + 8)
      var count = llama_tokenize(vocab, text, bytes, &tokens, Int32(tokens.count), true, false)
      if count < 0 {
        tokens = [llama_token](repeating: 0, count: Int(-count))
        count = llama_tokenize(vocab, text, bytes, &tokens, Int32(tokens.count), true, false)
      }
      guard count >= 0 else { throw TokenizeException() }
      let n = min(Int(count), size)
      tokenCount += n

      llama_memory_clear(llama_get_memory(context), true)
      var batch = llama_batch_init(Int32(n), 0, 1)
      defer { llama_batch_free(batch) }
      for i in 0..<n {
        batch.token[i] = tokens[i]
        batch.pos[i] = llama_pos(i)
        batch.n_seq_id[i] = 1
        batch.seq_id[i]![0] = 0
        batch.logits[i] = 1
      }
      batch.n_tokens = Int32(n)
      let status = encoder ? llama_encode(context, batch) : llama_decode(context, batch)
      guard status == 0 else { throw DecodeException(status) }
      guard let pooled = llama_get_embeddings_seq(context, 0) else { throw PoolingException() }
      var norm = 0.0
      for d in 0..<dimensions { norm += Double(pooled[d]) * Double(pooled[d]) }
      norm = norm > 0 ? norm.squareRoot() : 1
      vectors.append((0..<dimensions).map { Double(pooled[$0]) / norm })
    }
    return [
      "vectors": vectors,
      "tokens": tokenCount,
      "ms": milliseconds(started, DispatchTime.now()),
    ]
  }

  // The named session, made on first use with the load's context parameters and `size`
  // tokens of context (the load's `contextSize` when nil): the weights are shared, only the
  // KV cache is its own. Asked with another size, it is made again and its cache is lost.
  private func sessionNamed(_ name: String, size: Int?) throws -> Session {
    guard let model else { throw ModelNotLoadedException() }
    var params = contextParams
    if let size { params.n_ctx = UInt32(size) }
    if let session = sessions[name] {
      if session.size == Int(params.n_ctx) { return session }
      llama_free(session.context)
      sessions[name] = nil
    }
    guard let context = llama_init_from_model(model, params) else {
      throw ContextException(Int(params.n_ctx))
    }
    let session = Session(context, size: Int(params.n_ctx))
    sessions[name] = session
    return session
  }

  // Formats `messages` with the model's own chat template, then samples until an
  // end-of-generation token, `maxTokens`, a full context or `requestStop`. In the session
  // `options.session`, the part of the prompt its cache already holds is not read again.
  func generate(
    messages: [ChatMessage],
    options: GenerateOptions,
    onText: (String) -> Void
  ) throws -> [String: Any] {
    guard let model else { throw ModelNotLoadedException() }
    let session = try sessionNamed(options.session ?? "", size: options.contextSize)
    let context = session.context
    resetStop()

    let vocab = llama_model_get_vocab(model)
    let prompt = try applyTemplate(model: model, messages: messages)
    var tokens = try tokenize(vocab: vocab, text: prompt)

    let contextSize = Int(llama_n_ctx(context))
    guard tokens.count < contextSize else {
      throw PromptTooLongException((tokens.count, contextSize))
    }

    // The whole history comes in `messages`; what the cache holds of its start stays. At
    // least the last token is read again: its logits start the answer.
    let memory = llama_get_memory(context)
    var reused = 0
    let limit = min(session.cached.count, tokens.count - 1)
    while reused < limit, session.cached[reused] == tokens[reused] { reused += 1 }
    if reused == 0 || !llama_memory_seq_rm(memory, 0, llama_pos(reused), -1) {
      llama_memory_clear(memory, true)
      reused = 0
    }
    // Until the prompt is read, the cache holds only what is kept: an error leaves it true.
    session.cached = Array(tokens.prefix(reused))

    let promptStart = DispatchTime.now()
    let batchSize = Int(llama_n_batch(context))
    var offset = reused
    while offset < tokens.count {
      let count = min(batchSize, tokens.count - offset)
      let status = tokens.withUnsafeMutableBufferPointer { buffer in
        llama_decode(context, llama_batch_get_one(buffer.baseAddress! + offset, Int32(count)))
      }
      guard status == 0 else { throw DecodeException(status) }
      offset += count
    }
    session.cached = tokens
    let promptEnd = DispatchTime.now()

    let sampler = try makeSampler(options, vocab: vocab)
    defer { llama_sampler_free(sampler) }

    var text = ""
    var pending: [UInt8] = []
    var generated = 0
    var stopped = false

    while generated < options.maxTokens, tokens.count + generated < contextSize {
      if takeStop() {
        stopped = true
        break
      }
      var token = llama_sampler_sample(sampler, context, -1)
      if llama_vocab_is_eog(vocab, token) { break }

      pending += piece(vocab: vocab, token: token)
      // A multi-byte character can span tokens: emit only complete UTF-8.
      if let chunk = String(bytes: pending, encoding: .utf8) {
        text += chunk
        onText(chunk)
        pending.removeAll()
      } else if pending.count > 8 {
        let chunk = String(decoding: pending, as: UTF8.self)
        text += chunk
        onText(chunk)
        pending.removeAll()
      }

      generated += 1
      let status = llama_decode(context, llama_batch_get_one(&token, 1))
      guard status == 0 else { throw DecodeException(status) }
      session.cached.append(token)
    }
    if !pending.isEmpty {
      let chunk = String(decoding: pending, as: UTF8.self)
      text += chunk
      onText(chunk)
    }
    let end = DispatchTime.now()

    let promptMs = milliseconds(promptStart, promptEnd)
    let generationMs = milliseconds(promptEnd, end)
    return [
      "text": text,
      "promptTokens": tokens.count,
      "cachedTokens": reused,
      "tokens": generated,
      "stopped": stopped,
      "promptMs": promptMs,
      "generationMs": generationMs,
      "tokensPerSecond": generationMs > 0 ? Double(generated) * 1000 / generationMs : 0,
    ]
  }

  private func makeSampler(
    _ options: GenerateOptions, vocab: OpaquePointer?
  ) throws -> UnsafeMutablePointer<llama_sampler> {
    let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
    // First in the chain: tokens the grammar forbids never reach the other samplers.
    if let grammar = options.grammar, !grammar.isEmpty {
      guard let constraint = llama_sampler_init_grammar(vocab, grammar, "root") else {
        llama_sampler_free(chain)
        throw GrammarException()
      }
      llama_sampler_chain_add(chain, constraint)
    }
    if options.presencePenalty != 0 {
      // Last 64 tokens, as in llama.cpp's own defaults.
      llama_sampler_chain_add(
        chain,
        llama_sampler_init_penalties(
          llama_vocab_n_tokens(vocab), 64, 1, 0, Float(options.presencePenalty)))
    }
    if options.temperature <= 0 {
      llama_sampler_chain_add(chain, llama_sampler_init_greedy())
      return chain
    }
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(options.topK)))
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(Float(options.topP), 1))
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(Float(options.minP), 1))
    llama_sampler_chain_add(chain, llama_sampler_init_temp(Float(options.temperature)))
    let seed = options.seed.map { UInt32(truncatingIfNeeded: $0) } ?? UInt32.random(in: .min ... .max)
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
    return chain
  }

  private func applyTemplate(model: OpaquePointer, messages: [ChatMessage]) throws -> String {
    guard let template = llama_model_chat_template(model, nil) else {
      throw ChatTemplateException()
    }
    // strdup keeps the C strings alive for the whole call.
    let roles = messages.map { strdup($0.role) }
    let contents = messages.map { strdup($0.content) }
    defer {
      roles.forEach { free($0) }
      contents.forEach { free($0) }
    }
    let chat = zip(roles, contents).map { llama_chat_message(role: $0, content: $1) }

    var capacity = max(1024, messages.reduce(0) { $0 + $1.content.utf8.count } * 2)
    while true {
      var buffer = [CChar](repeating: 0, count: capacity)
      let length = llama_chat_apply_template(
        template, chat, chat.count, true, &buffer, Int32(capacity))
      guard length >= 0 else { throw ChatTemplateException() }
      if Int(length) <= capacity {
        return String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
      }
      capacity = Int(length)
    }
  }

  private func tokenize(vocab: OpaquePointer?, text: String) throws -> [llama_token] {
    let bytes = Int32(text.utf8.count)
    // The template already contains the special tokens, so parse them, don't add BOS twice.
    var tokens = [llama_token](repeating: 0, count: Int(bytes) + 8)
    var count = llama_tokenize(vocab, text, bytes, &tokens, Int32(tokens.count), false, true)
    if count < 0 {
      tokens = [llama_token](repeating: 0, count: Int(-count))
      count = llama_tokenize(vocab, text, bytes, &tokens, Int32(tokens.count), false, true)
    }
    guard count >= 0 else { throw TokenizeException() }
    return Array(tokens.prefix(Int(count)))
  }

  private func piece(vocab: OpaquePointer?, token: llama_token) -> [UInt8] {
    var buffer = [CChar](repeating: 0, count: 64)
    var length = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, false)
    if length < 0 {
      buffer = [CChar](repeating: 0, count: Int(-length))
      length = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, false)
    }
    return buffer.prefix(Int(max(0, length))).map { UInt8(bitPattern: $0) }
  }

  private func milliseconds(_ start: DispatchTime, _ end: DispatchTime) -> Double {
    Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
  }
}
