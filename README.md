# expo-llama

Run GGUF language models on iPhone, fully on the device. An Expo native module in Swift over
[llama.cpp](https://github.com/ggml-org/llama.cpp): load a model, stream an answer token by
token, stop it, unload it. Plus the file-side helpers a model download needs: SHA-256 of a
multi-gigabyte file, a reachability check that tells "offline" from "blocked", and exclusion
from iCloud backups.

Built for [OpenMycel](https://github.com/openmycel/openmycel), a private assistant that works
offline.

**Status:** early. iOS only. Sessions (0.3.0) are measured in the iOS Simulator only, not yet
on an iPhone — see [What it does not do](#what-it-does-not-do).

## Install

Needs Expo SDK 57+ with React Native 0.86+, iOS 16.4+, and a development build (native modules
do not run in Expo Go).

```sh
npx expo install @openmycel/expo-llama
npx expo prebuild   # or: cd ios && pod install
```

The llama.cpp engine ships inside the package as a ready `llama.xcframework` (8.7 MB download,
26 MB unpacked): nothing is compiled or downloaded during `pod install`.

## Usage

```typescript
import { generate, loadModel, stop, unload } from "@openmycel/expo-llama";

const info = await loadModel(fileUri, { contextSize: 2048 });
// { description: "qwen3 0.6B Q8_0", parameters: 596049920, sizeBytes: 633495552, contextSize: 2048 }

const result = await generate(
  [
    { role: "system", content: "Answer briefly." },
    { role: "user", content: "Two ideas for a quick dinner?" },
  ],
  { temperature: 0.7, topP: 0.8, topK: 20, minP: 0, presencePenalty: 1.5 },
  (piece) => console.log(piece) // streamed as it is generated
);
// { text, promptTokens, cachedTokens, tokens, stopped, promptMs, generationMs, tokensPerSecond }

stop(); // from anywhere: the running generate resolves with what it has so far
await unload();
```

The prompt is built with the model's own chat template (`llama_chat_apply_template`), so the
app passes plain `{ role, content }` messages. The module keeps no conversation: pass the whole
history each time; the session's cache saves reading it again ([Sessions](#sessions)).

## API

| Function                                | What it does                                                                                                                                                          |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `loadModel(path, options?)`             | Loads a GGUF file (path or `file://` URI), replacing the model loaded before. `contextSize` (default 4096), `gpuLayers` (default -1 = all on Metal).                  |
| `generate(messages, options?, onText?)` | Answers the conversation; `onText` gets each piece of text. Options below.                                                                                            |
| `stop()`                                | Ends a running `generate`; it resolves with `stopped: true`.                                                                                                          |
| `unload()` / `isLoaded()`               | Frees the model and every session / reports whether a model is loaded.                                                                                                |
| `sha256File(path)`                      | SHA-256 of a file as lowercase hex, read in 4 MB chunks with CryptoKit — a 3 GB model never sits in memory.                                                           |
| `cancelSha256File()`                    | Stops the running `sha256File`; it rejects with `HashCancelledException` within one chunk.                                                                            |
| `checkSource(url)`                      | One `HEAD` request (ephemeral session, no cookies) and why it failed: `offline`, `unreachable` (DNS, TLS, timeout, HTTP 403/451 — what a block looks like) or `http`. |
| `onMemoryWarning(listener)`             | iOS says memory is low: `unload()` now, before the app is killed. Returns the unsubscribe.                                                                            |
| `excludeFromBackup(path)`               | Keeps a re-downloadable model out of iCloud and device backups.                                                                                                       |

`generate` options; defaults are llama.cpp's own:

| Option            | Default | What it does                                                                         |
| ----------------- | ------- | ------------------------------------------------------------------------------------ |
| `maxTokens`       | 1024    | Stops after this many tokens.                                                        |
| `temperature`     | 0.8     | `0` = greedy.                                                                        |
| `topK`            | 40      |                                                                                      |
| `topP`            | 0.95    |                                                                                      |
| `minP`            | 0.05    |                                                                                      |
| `presencePenalty` | 0       | `0` = off; looks at the last 64 tokens.                                              |
| `seed`            | random  |                                                                                      |
| `grammar`         | —       | GBNF with a `root` rule: the answer can only be what it allows, e.g. one JSON shape. |
| `session`         | `""`    | The context to run in, made on first use ([Sessions](#sessions)).                    |
| `contextSize`     | load's  | The session's context size in tokens ([Sessions](#sessions)).                        |

The result: `text`, `promptTokens`, `cachedTokens` (of them, taken from the session's cache),
`tokens`, `stopped`, `promptMs`, `generationMs`, `tokensPerSecond`.

Errors are thrown as typed exceptions with a readable message: `ModelNotFoundException`,
`ModelLoadException`, `ContextException`, `ModelNotLoadedException`, `ChatTemplateException`,
`PromptTooLongException`, `DecodeException`, `HashCancelledException`.

## Sessions

A session is a context of its own on the loaded model: the weights are shared, the KV cache
is its own. It keeps what it last read, so a prompt that starts the same way as the one before
— the same system prompt, a chat one message longer — is read only from where it differs.
`cachedTokens` in the result says how much was skipped.

Use one session per kind of prompt, so they do not overwrite each other's cache: the default
`""` for the chat, `"router"` for a fixed instruction asked again and again. Calls in different
sessions still run one at a time, on the same queue.

Each session costs memory: its KV cache is allocated for its whole context when the session
is first used — for an f16 cache, `2 × layers × KV heads × head size × 2 bytes` per token.
By default that is the `contextSize` of `loadModel`; a session that only reads short prompts
can ask for less with the `contextSize` option of `generate`. Asked with another size than it
has, the session is made again and its cache is lost. `unload()` frees all sessions; there is
no way to free one.

## What it does not do

- **Android.** iOS only; Android is planned in the same module.
- **Freeing one session.** Only `unload()`, which frees the model and all of them.
- **Two generations at once.** One queue: a second `generate` waits for the first.
- **Keeping the cache across launches.** A session's cache lives in memory until `unload()`.
- **Measured on an iPhone (0.3.0).** Sessions and the prompt cache are checked in the iOS
  Simulator, on the CPU. Speed and memory with Metal on a device are not measured yet.
- **Embeddings, images, audio, LoRA adapters, tool calls.** Text in, text out; `grammar` is
  the only way to shape the answer.
- **Downloading models.** The app downloads the file; the module checks it (`sha256File`,
  `checkSource`) and keeps it out of backups.

## Architecture

```text
 JavaScript (the app)
   src/index.ts ── thin wrappers; subscribes to "onToken" for the length of one generate
        │  Expo Modules (JSI)
 Swift ─┼──────────────────────────────────────────────────────────────────────────
   ExpoLlamaModule.swift ── definitions, argument records, typed exceptions
        │                    stop() is a plain Function: it must reach a running generate
        ▼
   LlamaEngine.swift ── one model + a context per session; every call on one serial queue
        │   load:     llama_model_load_from_file → llama_init_from_model (session "")
        │   generate: chat template → tokenize → keep the cached prefix (llama_memory_seq_rm)
        │             → decode the rest of the prompt in n_batch chunks
        │             → sample (penalties → top-k → top-p → min-p → temperature → dist)
        │             → emit complete UTF-8 only → decode the token → repeat
        │   stop:     a lock-guarded flag checked before every token
        ▼
   llama.xcframework (llama.cpp b11146, built from source) ── Metal on device, CPU in the simulator

   ModelFiles.swift ── sha256 (CryptoKit), excludeFromBackup, checkSource (URLSession HEAD)
```

Decisions worth knowing:

- **One serial queue.** Load, generate and unload never overlap, so the engine needs no locks
  around llama.cpp state. Only `stop` crosses threads.
- **UTF-8 at token boundaries.** A multi-byte character can be split across tokens; bytes are
  held until they form valid UTF-8, so the app never renders a broken character.
- **CPU in the simulator.** With Metal in the iOS Simulator llama.cpp produced garbage (only
  `!` tokens, Xcode 27, b11146); the CPU gives correct output. `gpuLayers` is ignored there.
  Real speed can only be measured on an iPhone.
- **Quiet logs.** llama.cpp logs every graph node at load; the module keeps warnings and
  errors only.
- **No network in the engine.** The only request the module can make is `checkSource`, and
  only when the app calls it. No telemetry.

## Building the engine

llama.cpp publishes a prebuilt `llama-bNNNNN-xcframework.zip` with every build, but it holds
iPhone and macOS only — no iOS Simulator slice. So the module builds its own from source:

```sh
npm run build:llama   # scripts/build-llama.sh
```

The script downloads the pinned tag (`b11146`, the build behind the `v0.5.0` release),
refuses the archive unless `git get-tar-commit-id` matches the pinned commit, and runs
llama.cpp's own `build-xcframework.sh ios-sim ios-device`. Needs Xcode and CMake; on an M3 Max
it takes about 90 seconds.

Output: `ios/Frameworks/llama.xcframework` (iPhone arm64 + simulator arm64/x86_64) — in the npm
package, not in git. Debug symbols make up ~170 of ~195 MB, so they are split off into
`dist/llama-<tag>-dSYMs.zip` for the GitHub release, where crash reports can be symbolicated.

## Developing the module

Link it next to an app:

```sh
npm install ../expo-llama        # "file:" dependency, a symlink
cd ios && pod install            # autolinking picks up ExpoLlama.podspec
```

Metro has to watch the module's folder and resolve `expo` and `react-native` from the app, so
the bundle holds one copy of each; TypeScript needs `preserveSymlinks`. See
`apps/mobile/metro.config.js` and `tsconfig.json` in OpenMycel. New Swift files need another
`pod install`.

## Credits

[llama.cpp](https://github.com/ggml-org/llama.cpp) by the ggml authors, MIT; the package ships
its license next to the framework, `ios/Frameworks/LICENSE-llama.cpp`. Module layout from
Expo's `expo-module-template`, MIT.

## Author and license

Created by Nikita MRCS — [@nmrcs](https://github.com/nmrcs). Contributors are
credited in the git history.

MIT.
