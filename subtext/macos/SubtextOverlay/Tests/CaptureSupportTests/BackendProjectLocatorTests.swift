import Foundation
import XCTest
@testable import CaptureSupport

final class BackendProjectLocatorTests: XCTestCase {
    func testFindsBackendFromNestedAppBundlePath() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("SubtextBackendLocator-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let backend = root.appendingPathComponent("backend", isDirectory: true)
        let python = backend.appendingPathComponent(".venv/bin/python")
        let app = backend.appendingPathComponent("app.py")
        let bundle = root
            .appendingPathComponent("macos/SubtextOverlay/.build/Products/Release/SubtextOverlay.app", isDirectory: true)

        try FileManager.default.createDirectory(at: python.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        XCTAssertTrue(FileManager.default.createFile(atPath: python.path, contents: Data("#!/bin/sh\n".utf8)))
        try Data("# backend entry point\n".utf8).write(to: app)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: python.path)

        XCTAssertEqual(
            BackendProjectLocator.backendDirectory(bundleURL: bundle, environment: [:]),
            backend
        )
    }

    func testConfiguredBackendDirectoryTakesPriority() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("SubtextBackendLocator-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let backend = root.appendingPathComponent("service", isDirectory: true)
        let python = backend.appendingPathComponent(".venv/bin/python")
        try FileManager.default.createDirectory(at: python.deletingLastPathComponent(), withIntermediateDirectories: true)
        XCTAssertTrue(FileManager.default.createFile(atPath: python.path, contents: Data("#!/bin/sh\n".utf8)))
        try Data("# backend entry point\n".utf8).write(to: backend.appendingPathComponent("app.py"))
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: python.path)

        XCTAssertEqual(
            BackendProjectLocator.backendDirectory(
                bundleURL: URL(fileURLWithPath: "/unrelated/SubtextOverlay.app"),
                environment: ["SUBTEXT_BACKEND_DIR": backend.path]
            ),
            backend
        )
    }
}
