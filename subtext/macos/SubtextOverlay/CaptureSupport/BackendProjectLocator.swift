import Foundation

public enum BackendProjectLocator {
    public static func backendDirectory(
        bundleURL: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> URL? {
        var candidates: [URL] = []

        if let backendPath = environment["SUBTEXT_BACKEND_DIR"], !backendPath.isEmpty {
            candidates.append(URL(fileURLWithPath: backendPath, isDirectory: true))
        }
        if let projectPath = environment["SUBTEXT_PROJECT_ROOT"], !projectPath.isEmpty {
            candidates.append(
                URL(fileURLWithPath: projectPath, isDirectory: true)
                    .appendingPathComponent("backend", isDirectory: true)
            )
        }

        var directory = bundleURL.standardizedFileURL.deletingLastPathComponent()
        for _ in 0..<12 {
            candidates.append(directory.appendingPathComponent("backend", isDirectory: true))
            directory.deleteLastPathComponent()
        }

        return candidates.first { isUsableBackend(at: $0, fileManager: fileManager) }
    }

    private static func isUsableBackend(at directory: URL, fileManager: FileManager) -> Bool {
        let python = directory.appendingPathComponent(".venv/bin/python")
        let app = directory.appendingPathComponent("app.py")
        return fileManager.isExecutableFile(atPath: python.path)
            && fileManager.fileExists(atPath: app.path)
    }
}
