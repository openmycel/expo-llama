import CryptoKit
import Foundation

// File-side helpers for model downloads: integrity, backup and source checks.
enum ModelFiles {
  // Streams the file through SHA-256 in 4 MB chunks, so a 3 GB model never sits in memory.
  // `isCancelled` is asked before every chunk: a hash of a 3 GB file stops within one chunk.
  // Each chunk is read inside its own autorelease pool: without it the chunks pile up until
  // the loop ends — 1.7 GB for a 1.8 GB file on a background queue (measured), enough to be
  // killed on a phone with the model loaded.
  static func sha256(path: String, isCancelled: () -> Bool = { false }) throws -> String {
    guard let handle = FileHandle(forReadingAtPath: path) else {
      throw ModelNotFoundException(path)
    }
    defer { try? handle.close() }
    var hasher = SHA256()
    while true {
      if isCancelled() { throw HashCancelledException() }
      let more = try autoreleasepool { () throws -> Bool in
        guard let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty else {
          return false
        }
        hasher.update(data: chunk)
        return true
      }
      if !more { break }
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  // Models can be downloaded again, so they stay out of iCloud and device backups.
  static func excludeFromBackup(path: String) throws {
    var url = URL(fileURLWithPath: path)
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try url.setResourceValues(values)
  }

  // One HEAD request to the download URL (redirects followed), no cookies or cache.
  // `reason` tells the app what to say: the phone is offline, the host cannot be
  // reached (DNS, TLS, timeouts — what a block looks like), or the server said no.
  static func checkSource(url: URL, completion: @escaping ([String: Any]) -> Void) {
    let config = URLSessionConfiguration.ephemeral
    config.timeoutIntervalForRequest = 15
    config.waitsForConnectivity = false
    let session = URLSession(configuration: config)
    var request = URLRequest(url: url)
    request.httpMethod = "HEAD"

    session.dataTask(with: request) { _, response, error in
      defer { session.finishTasksAndInvalidate() }
      if let error = error as? URLError {
        completion(["ok": false, "reason": reason(for: error.code), "code": error.code.rawValue])
        return
      }
      if let error {
        completion(["ok": false, "reason": "unreachable", "message": error.localizedDescription])
        return
      }
      let status = (response as? HTTPURLResponse)?.statusCode ?? 0
      if (200...299).contains(status) {
        completion(["ok": true, "status": status])
      } else {
        // 403 and 451 are how a server-side block usually answers.
        let blocked = status == 403 || status == 451
        completion(["ok": false, "reason": blocked ? "unreachable" : "http", "status": status])
      }
    }.resume()
  }

  private static func reason(for code: URLError.Code) -> String {
    switch code {
    case .notConnectedToInternet, .dataNotAllowed, .internationalRoamingOff, .callIsActive:
      return "offline"
    default:
      return "unreachable"
    }
  }
}
