import AppKit
import Combine
import Foundation
import SwiftUI
import ScreenCaptureKit
import AVFoundation
import CoreMedia
import CoreVideo
import CoreImage
import AudioToolbox
import CoreAudio
import ImageIO
import Vision
import CaptureSupport

@main
struct SubtextOverlayApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let client = TranscriptionClient()
    private var panel: NSPanel?
    private var speakerOverlayPanel: NSPanel?
    private var speakerOverlayFrame: CGRect?
    private var speakerOverlayProcessID: pid_t?
    private var speakerOverlayWindowID: UInt32?
    private var hasSpeakerTile = false
    private var workspaceActivationObserver: NSObjectProtocol?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 390, height: 720),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.title = "Subtext"
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true
        panel.isMovableByWindowBackground = true
        panel.contentView = NSHostingView(rootView: OverlayView(client: client))

        if let screen = NSScreen.main {
            let frame = screen.visibleFrame
            panel.setFrameOrigin(
                NSPoint(
                    x: frame.maxX - panel.frame.width - 24,
                    y: frame.maxY - panel.frame.height - 36
                )
            )
        }

        self.panel = panel
        panel.orderFrontRegardless()

        let speakerOverlayPanel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 640, height: 360),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        speakerOverlayPanel.title = "Speaker tile glow"
        speakerOverlayPanel.level = .floating
        speakerOverlayPanel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        speakerOverlayPanel.backgroundColor = .clear
        speakerOverlayPanel.isOpaque = false
        speakerOverlayPanel.hasShadow = false
        speakerOverlayPanel.ignoresMouseEvents = true
        speakerOverlayPanel.hidesOnDeactivate = false
        speakerOverlayPanel.contentView = NSHostingView(rootView: SpeakerTileOverlayView(client: client))
        self.speakerOverlayPanel = speakerOverlayPanel
        client.speakerOverlayUpdateHandler = { [weak self] frame, processID, windowID, hasTile in
            self?.updateSpeakerOverlay(frame: frame, processID: processID, windowID: windowID, hasTile: hasTile)
        }
        workspaceActivationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.syncSpeakerOverlayVisibility() }
        }

        NSApp.activate(ignoringOtherApps: true)
        client.connect()
    }

    private func updateSpeakerOverlay(frame: CGRect?, processID: pid_t?, windowID: UInt32?, hasTile: Bool) {
        speakerOverlayFrame = frame
        speakerOverlayProcessID = processID
        speakerOverlayWindowID = windowID
        hasSpeakerTile = hasTile
        syncSpeakerOverlayVisibility()
    }

    private func syncSpeakerOverlayVisibility() {
        guard let frame = speakerOverlayFrame,
              let processID = speakerOverlayProcessID,
              let windowID = speakerOverlayWindowID,
              hasSpeakerTile,
              NSWorkspace.shared.frontmostApplication?.processIdentifier == processID,
              Self.frontmostWindowID(for: processID) == windowID,
              let speakerOverlayPanel else {
            speakerOverlayPanel?.orderOut(nil)
            return
        }
        if speakerOverlayPanel.frame != frame {
            speakerOverlayPanel.setFrame(frame, display: false)
        }
        if !speakerOverlayPanel.isVisible {
            speakerOverlayPanel.orderFrontRegardless()
        }
    }

    private static func frontmostWindowID(for processID: pid_t) -> UInt32? {
        guard let windows = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] else {
            return nil
        }
        for window in windows {
            guard let ownerPID = window[kCGWindowOwnerPID as String] as? NSNumber,
                  ownerPID.int32Value == processID,
                  let layer = window[kCGWindowLayer as String] as? NSNumber,
                  layer.intValue == 0,
                  let bounds = window[kCGWindowBounds as String] as? [String: Any],
                  let width = (bounds["Width"] as? NSNumber)?.doubleValue,
                  let height = (bounds["Height"] as? NSNumber)?.doubleValue,
                  width >= 200,
                  height >= 200,
                  let windowNumber = window[kCGWindowNumber as String] as? NSNumber else {
                continue
            }
            return windowNumber.uint32Value
        }
        return nil
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let workspaceActivationObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(workspaceActivationObserver)
        }
        speakerOverlayPanel?.orderOut(nil)
        client.stopOwnedBackend()
    }
}

struct TranscriptEntry: Identifiable {
    let id: String
    let text: String
    let timestamp: Date
    let speakerID: String
}

struct FaceLandmarkPreview: Sendable {
    let boundingBox: [Double]
    let points: [[Double]]
}

struct LandmarkPreviewFrame: Sendable {
    let jpegData: Data
    let faces: [FaceLandmarkPreview]
    var speakerTileBox: [Double]?
    let speakerFaceIndex: Int?
    var speakerTrackID: String?
    let imageAspectRatio: Double
}

private struct SpeakerTileCandidate {
    let bounds: CGRect
    let faceIndex: Int
    let score: Double
}

private func normalizedBox(from rect: CGRect) -> [Double] {
    [Double(rect.minX), Double(rect.minY), Double(rect.width), Double(rect.height)]
}

private func normalizedRect(from box: [Double]?) -> CGRect? {
    guard let box, box.count == 4,
          box.allSatisfy(\.isFinite),
          box[2] > 0, box[3] > 0 else {
        return nil
    }
    let rect = CGRect(x: CGFloat(box[0]), y: CGFloat(box[1]), width: CGFloat(box[2]), height: CGFloat(box[3]))
        .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    return rect.isNull || rect.isEmpty ? nil : rect
}

struct AnalysisObservation: Decodable {
    let observation: String
    let evidenceIds: [String]
}

struct AnalysisHypothesis: Decodable {
    let interpretation: String
    let evidenceIds: [String]
    let alternatives: [String]
    let confidence: String
    let possibleCheckIn: String?
}

struct ImpliedMeaningSignal: Decodable {
    let detected: Bool
    let kind: String
    let quote: String
    let meaning: String
    let evidenceIds: [String]
    let confidence: Int
}

struct ConversationAnalysis: Decodable {
    let summary: String
    let notableObservations: [AnalysisObservation]
    let hypotheses: [AnalysisHypothesis]
    let noClearSignal: Bool
    let impliedMeaning: ImpliedMeaningSignal?
}

private struct ValenceScorePayload: Decodable {
    let trackId: String
    let valence: Double?
    let framesSeen: Int
}

struct ValenceEstimate: Identifiable {
    let trackID: String
    let score: Double?
    let framesSeen: Int

    var id: String { trackID }
    var personLabel: String {
        let suffix = trackID.split(separator: "_").last ?? "1"
        return "Person \(suffix)"
    }
}

private struct ServerEvent: Decodable {
    let type: String
    let state: String?
    let detail: String?
    let id: String?
    let text: String?
    let transcriptId: String?
    let timestamp: Double?
    let speakerId: String?
    let trigger: String?
    let outcome: String?
    let geminiConfigured: Bool?
    let analysis: ConversationAnalysis?
    let scores: [ValenceScorePayload]?
    let quote: String?
    let interpretation: String?
}

private struct QueuedCaptureMessage {
    let type: String
    let text: String
}

enum MismatchCheckStatus: Equatable {
    case checking
    case confirmed
    case unclear
    case unavailable
    case needsSetup
    case needsVisualSource

    var label: String {
        switch self {
        case .checking: "Reviewing"
        case .confirmed: "Possible mismatch"
        case .unclear: "Unclear"
        case .unavailable: "Review unavailable"
        case .needsSetup: "Local cue"
        case .needsVisualSource: "Local cue"
        }
    }

    var detail: String {
        switch self {
        case .checking: "Words and voice raised a cue · checking visual context"
        case .confirmed: ""
        case .unclear: "No clear mismatch was confirmed."
        case .unavailable: "The multimodal review could not finish."
        case .needsSetup: "Words and voice raised a cue · add a Gemini key for review."
        case .needsVisualSource: "Words and voice raised a cue · choose a call window for visual review."
        }
    }

    var visualColor: Color {
        switch self {
        case .checking: .orange
        case .confirmed: .green
        case .unclear, .needsSetup, .needsVisualSource: .white.opacity(0.48)
        case .unavailable: .orange.opacity(0.72)
        }
    }
}

struct MismatchCheck: Identifiable {
    let id: String
    let transcriptID: String?
    var quote: String
    var interpretation: String?
    var status: MismatchCheckStatus
}

@MainActor
private final class LocalBackendSupervisor {
    private enum StartupError: LocalizedError {
        case projectNotFound
        case launchFailed(String)
        case didNotBecomeReady

        var errorDescription: String? {
            switch self {
            case .projectNotFound:
                return "The local transcription service is not running, and Subtext could not find its Python environment. Launch Subtext with scripts/run.sh once to install it, then reopen the app."
            case .launchFailed(let detail):
                return "Subtext could not start its local transcription service: \(detail)"
            case .didNotBecomeReady:
                return "The local transcription service did not become ready. Check the project’s scripts/run.sh output for startup errors."
            }
        }
    }

    private let serviceURL: URL
    private var ownedProcess: Process?

    init(serviceURL: URL) {
        self.serviceURL = serviceURL
    }

    func ensureAvailable() async throws {
        if await serviceIsHealthy() {
            return
        }

        if let ownedProcess, ownedProcess.isRunning {
            do {
                try await waitForService()
            } catch {
                stopOwnedProcess()
                throw error
            }
            return
        }
        ownedProcess = nil

        guard let backendDirectory = BackendProjectLocator.backendDirectory(
            bundleURL: Bundle.main.bundleURL
        ) else {
            throw StartupError.projectNotFound
        }

        let python = backendDirectory.appendingPathComponent(".venv/bin/python")
        let process = Process()
        process.executableURL = python
        process.arguments = [
            "-m", "uvicorn", "app:app",
            "--host", "127.0.0.1",
            "--port", "8765",
            "--log-level", "info"
        ]
        process.currentDirectoryURL = backendDirectory
        process.environment = backendEnvironment(at: backendDirectory)
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
        } catch {
            throw StartupError.launchFailed(error.localizedDescription)
        }
        ownedProcess = process

        do {
            try await waitForService()
        } catch {
            stopOwnedProcess()
            throw error
        }
    }

    func stopOwnedProcess() {
        guard let process = ownedProcess else { return }
        ownedProcess = nil
        if process.isRunning {
            process.terminate()
        }
    }

    private func waitForService() async throws {
        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            if Task.isCancelled {
                throw CancellationError()
            }
            if await serviceIsHealthy() {
                return
            }
            if ownedProcess?.isRunning == false {
                ownedProcess = nil
                throw StartupError.didNotBecomeReady
            }
            try await Task.sleep(nanoseconds: 250_000_000)
        }
        throw StartupError.didNotBecomeReady
    }

    private func serviceIsHealthy() async -> Bool {
        var request = URLRequest(url: serviceURL.appending(path: "/health"))
        request.timeoutInterval = 1
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              let response = response as? HTTPURLResponse,
              (200..<300).contains(response.statusCode),
              let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        return body["ok"] as? Bool == true
    }

    private func backendEnvironment(at backendDirectory: URL) -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        let envFile = backendDirectory.appendingPathComponent(".env.local")
        guard let contents = try? String(contentsOf: envFile, encoding: .utf8) else {
            return environment
        }

        for rawLine in contents.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, !line.hasPrefix("#"), let separator = line.firstIndex(of: "=") else {
                continue
            }
            let key = line[..<separator].trimmingCharacters(in: .whitespaces)
            guard key.range(of: "^[A-Za-z_][A-Za-z0-9_]*$", options: .regularExpression) != nil else {
                continue
            }
            if let existingValue = environment[key], !existingValue.isEmpty {
                continue
            }

            var value = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespaces)
            if value.count >= 2,
               let first = value.first,
               (first == "'" || first == "\""),
               value.last == first {
                value.removeFirst()
                value.removeLast()
            }
            environment[key] = value
        }
        return environment
    }
}

@MainActor
final class TranscriptionClient: ObservableObject {
    @Published private(set) var transcript: [TranscriptEntry] = []
    @Published private(set) var stateLabel = "Connecting"
    @Published private(set) var isConnected = false
    @Published private(set) var isListening = false
    @Published private(set) var isStarting = false
    @Published private(set) var microphoneLevel = 0.0
    @Published private(set) var microphoneBufferReceived = false
    @Published private(set) var microphoneMuted = true
    @Published private(set) var errorMessage: String?
    @Published private(set) var windows: [ShareableWindow] = []
    @Published private(set) var landmarkPreview: LandmarkPreviewFrame?
    @Published var selectedWindowID: UInt32?
    @Published private(set) var geminiConfigured = false
    @Published private(set) var latestAnalysis: ConversationAnalysis?
    @Published private(set) var latestMismatchCheck: MismatchCheck?
    @Published private(set) var valenceEstimates: [ValenceEstimate] = []
    @Published private(set) var valencePersonNames: [String: String] = [:]
    @Published private(set) var valenceModelState = "not_loaded"
    @Published private(set) var valenceModelDetail: String?

    private var latestAnalysisID: String?
    private var latestAnalysisTask: Task<Void, Never>?
    private var latestMismatchTask: Task<Void, Never>?
    private var receiveTask: Task<Void, Never>?
    private var socket: URLSessionWebSocketTask?
    private var ingestSocket: URLSessionWebSocketTask?
    private var captureSendTask: Task<Void, Never>?
    private var ingestReceiveTask: Task<Void, Never>?
    private var diagnosticsTask: Task<Void, Never>?
    private var captureMessageQueue: [QueuedCaptureMessage] = []
    private let maximumCaptureQueueDepth = 80
    private var captureQueueDroppedTotal = 0
    private var capturePacketCounts: [String: Int] = [:]
    private var microphoneBufferCount = 0
    private var captureWindowScreenFrame: CGRect?
    private var captureWindowProcessID: pid_t?
    private var captureWindowID: UInt32?
    private let microphoneCapture = MicrophoneCaptureCoordinator()
    private let captureCoordinator = ScreenCaptureCoordinator()
    private let serviceURL = URL(string: "http://127.0.0.1:8765")!
    private lazy var backendSupervisor = LocalBackendSupervisor(serviceURL: serviceURL)
    private let websocketSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        // Capture and transcript sockets can remain quiet in one direction
        // during long meetings. Keep URLSession's request timeout from ending
        // those long-lived tasks; the ingest socket also reads server frames
        // so WebSocket keepalive pings are serviced.
        configuration.timeoutIntervalForRequest = 24 * 60 * 60
        configuration.timeoutIntervalForResource = 24 * 60 * 60
        return URLSession(configuration: configuration)
    }()
    var speakerOverlayUpdateHandler: ((CGRect?, pid_t?, UInt32?, Bool) -> Void)?

    init() {
        microphoneCapture.messageHandler = { [weak self] message in
            Task { @MainActor in
                self?.sendCaptureMessage(message)
            }
        }
        microphoneCapture.levelHandler = { [weak self] level in
            Task { @MainActor in
                self?.microphoneLevel = level
            }
        }
        microphoneCapture.bufferHandler = { [weak self] in
            Task { @MainActor in
                self?.microphoneBufferReceived = true
                self?.microphoneBufferCount += 1
            }
        }
        microphoneCapture.inputStalledHandler = { [weak self] in
            Task { @MainActor in
                self?.microphoneBufferReceived = false
            }
        }
        microphoneCapture.diagnosticHandler = { [weak self] event, detail in
            Task { @MainActor in
                self?.reportDiagnostic(
                    component: "microphone_capture",
                    event: event,
                    detail: detail
                )
            }
        }
        captureCoordinator.messageHandler = { [weak self] message in
            self?.sendCaptureMessage(message)
        }
        captureCoordinator.personNameUpdateHandler = { [weak self] trackID, name in
            self?.valencePersonNames[trackID] = name
        }
        captureCoordinator.previewHandler = { [weak self] frame in
            guard let self else { return }
            self.landmarkPreview = frame
            self.updateSpeakerOverlay(for: frame)
        }
        captureCoordinator.windowFrameHandler = { [weak self] frame, processID, windowID in
            guard let self else { return }
            self.captureWindowScreenFrame = frame
            self.captureWindowProcessID = processID
            self.captureWindowID = windowID
            self.updateSpeakerOverlay(for: self.landmarkPreview)
        }
        captureCoordinator.captureErrorHandler = { [weak self] detail in
            self?.handleCaptureFailure(detail)
        }
    }

    private func updateSpeakerOverlay(for frame: LandmarkPreviewFrame?) {
        let hasTile = frame?.speakerTileBox != nil
        speakerOverlayUpdateHandler?(captureWindowScreenFrame, captureWindowProcessID, captureWindowID, hasTile)
    }

    func connect() {
        guard receiveTask == nil else { return }
        receiveTask = Task { [weak self] in
            guard let self else { return }

            while !Task.isCancelled {
                do {
                    try await backendSupervisor.ensureAvailable()
                } catch {
                    isConnected = false
                    isListening = false
                    stateLabel = "Service unavailable"
                    errorMessage = error.localizedDescription
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                    continue
                }

                let candidate = websocketSession.webSocketTask(
                    with: URL(string: "ws://127.0.0.1:8765/ws")!
                )
                socket = candidate
                candidate.resume()
                stateLabel = "Connecting"

                do {
                    while !Task.isCancelled {
                        let message = try await candidate.receive()
                        isConnected = true
                        handle(message)
                    }
                } catch {
                    isConnected = false
                    isListening = false
                    stateLabel = "Service disconnected"
                    if !Task.isCancelled {
                        reportDiagnostic(
                            component: "overlay_events_socket",
                            event: "receive_failed",
                            detail: error.localizedDescription
                        )
                    }
                    candidate.cancel(with: .goingAway, reason: nil)
                }

                guard !Task.isCancelled else { break }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    func start() async {
        errorMessage = nil
        landmarkPreview = nil
        microphoneLevel = 0
        microphoneBufferReceived = false
        capturePacketCounts.removeAll()
        captureQueueDroppedTotal = 0
        microphoneBufferCount = 0
        stateLabel = "Starting"
        let windowID = selectedWindowID

        // Starting a capture session is the user's explicit request to
        // transcribe both sides. The mic control remains available to mute
        // local transcription immediately at any point during the session.
        microphoneMuted = false
        microphoneCapture.setMuted(false)
        isStarting = true
        let captureSocket = websocketSession.webSocketTask(
            with: URL(string: "ws://127.0.0.1:8765/ingest")!
        )
        ingestSocket = captureSocket
        captureSocket.resume()
        startIngestReceiveLoop(for: captureSocket)
        do {
            try await post(path: "/start")
            sendCaptureMessage(["type": "microphone_gate", "enabled": true])
            try await microphoneCapture.start()
            if let windowID {
                try await captureCoordinator.start(windowID: windowID)
            } else {
                reportDiagnostic(
                    component: "screen_capture",
                    event: "microphone_only_session",
                    detail: "Microphone transcription started without a selected call window."
                )
            }
            startDiagnosticsHeartbeat()
            isStarting = false
            if windowID == nil {
                stateLabel = "Listening · microphone only"
            } else if !geminiConfigured {
                stateLabel = "Listening · Gemini key needed"
            }
        } catch {
            diagnosticsTask?.cancel()
            diagnosticsTask = nil
            reportDiagnostic(
                component: "capture_start",
                event: "start_failed",
                detail: error.localizedDescription,
                metadata: ["selected_window_id": windowID.map(String.init) ?? "none"]
            )
            isStarting = false
            isListening = false
            microphoneCapture.stop()
            microphoneCapture.setMuted(true)
            microphoneMuted = true
            try? await captureCoordinator.stop()
            ingestReceiveTask?.cancel()
            ingestReceiveTask = nil
            captureSocket.cancel(with: .goingAway, reason: nil)
            ingestSocket = nil
            try? await post(path: "/stop")
            errorMessage = error.localizedDescription
            stateLabel = "Could not start"
        }
    }

    func stop() async {
        diagnosticsTask?.cancel()
        diagnosticsTask = nil
        isStarting = false
        errorMessage = nil
        landmarkPreview = nil
        microphoneLevel = 0
        microphoneBufferReceived = false
        microphoneCapture.stop()
        microphoneCapture.setMuted(true)
        microphoneMuted = true
        valenceEstimates.removeAll()
        valencePersonNames.removeAll()
        do {
            try await captureCoordinator.stop()
            try await post(path: "/stop")
            ingestReceiveTask?.cancel()
            ingestReceiveTask = nil
            ingestSocket?.cancel(with: .goingAway, reason: nil)
            ingestSocket = nil
            captureMessageQueue.removeAll()
        } catch {
            reportDiagnostic(
                component: "capture_stop",
                event: "stop_failed",
                detail: error.localizedDescription
            )
            errorMessage = error.localizedDescription
            stateLabel = "Could not stop"
        }
    }

    func refreshWindows() async {
        do {
            try await captureCoordinator.refreshWindows()
            windows = captureCoordinator.windows
            errorMessage = windows.isEmpty
                ? "No open app windows to choose. Open the call window and refresh the list."
                : nil
            if let selectedWindowID, !windows.contains(where: { $0.id == selectedWindowID }) {
                self.selectedWindowID = nil
            }
            if selectedWindowID == nil, windows.count == 1 {
                selectedWindowID = windows[0].id
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func toggleMicrophoneMuted() {
        guard isListening else { return }
        microphoneMuted.toggle()
        microphoneCapture.setMuted(microphoneMuted)
        sendCaptureMessage([
            "type": "microphone_gate",
            "enabled": !microphoneMuted
        ])
        reportDiagnostic(
            component: "microphone_capture",
            event: microphoneMuted ? "transcription_muted" : "transcription_unmuted",
            detail: microphoneMuted
                ? "Local microphone transcription paused; call-window audio remains active."
                : "Local microphone transcription resumed."
        )
    }

    func clearTranscript() {
        transcript.removeAll()
        latestAnalysis = nil
        latestAnalysisID = nil
        latestAnalysisTask?.cancel()
        latestMismatchCheck = nil
        latestMismatchTask?.cancel()
        Task {
            var request = URLRequest(url: serviceURL.appending(path: "/transcript"))
            request.httpMethod = "DELETE"
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    func stopOwnedBackend() {
        backendSupervisor.stopOwnedProcess()
    }

    private func post(path: String) async throws {
        var request = URLRequest(url: serviceURL.appending(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = 240
        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw ClientError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let detail = (try? JSONDecoder().decode(ServiceError.self, from: data).detail)
                ?? "The local transcription service returned an error."
            throw ClientError.service(detail)
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        let data: Data
        switch message {
        case .data(let received):
            data = received
        case .string(let received):
            data = Data(received.utf8)
        @unknown default:
            return
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let event = try? decoder.decode(ServerEvent.self, from: data) else {
            return
        }

        if let geminiConfigured = event.geminiConfigured {
            self.geminiConfigured = geminiConfigured
        }

        if event.type == "cleared" {
            transcript.removeAll()
            latestAnalysis = nil
            latestAnalysisID = nil
            latestMismatchCheck = nil
            valenceEstimates.removeAll()
            valencePersonNames.removeAll()
            valenceModelState = "not_loaded"
            valenceModelDetail = nil
            latestAnalysisTask?.cancel()
            latestMismatchTask?.cancel()
            return
        }

        if event.type == "valence_cleared" {
            valenceEstimates.removeAll()
            valencePersonNames.removeAll()
            return
        }

        if event.type == "valence_status", let state = event.state {
            valenceModelState = state
            valenceModelDetail = event.detail
            return
        }

        if event.type == "valence_update" {
            valenceModelState = "ready"
            valenceModelDetail = nil
            valenceEstimates = (event.scores ?? []).map {
                ValenceEstimate(
                    trackID: $0.trackId,
                    score: $0.valence,
                    framesSeen: $0.framesSeen
                )
            }
            return
        }

        if event.type == "transcript", let text = event.text, !text.isEmpty {
            let entry = TranscriptEntry(
                id: event.id ?? UUID().uuidString,
                text: text,
                timestamp: Date(timeIntervalSince1970: event.timestamp ?? Date().timeIntervalSince1970),
                speakerID: event.speakerId ?? "self"
            )
            transcript.append(entry)
            if var check = latestMismatchCheck, check.transcriptID == entry.id {
                check.quote = entry.text
                latestMismatchCheck = check
            }
            if transcript.count > 40 {
                transcript.removeFirst(transcript.count - 40)
            }
            return
        }

        if event.type == "mismatch_candidate",
           let checkID = event.id,
           let transcriptID = event.transcriptId {
            latestMismatchTask?.cancel()
            latestMismatchCheck = MismatchCheck(
                id: checkID,
                transcriptID: transcriptID,
                quote: transcript.first(where: { $0.id == transcriptID })?.text ?? "",
                interpretation: nil,
                status: !geminiConfigured
                    ? .needsSetup
                    : (selectedWindowID == nil ? .needsVisualSource : .checking)
            )
            if !geminiConfigured || selectedWindowID == nil {
                scheduleMismatchCheckDismissal(checkID)
            }
            return
        }

        if event.type == "llm_analysis", let analysis = event.analysis {
            latestAnalysis = analysis
            latestAnalysisID = event.id
            latestAnalysisTask?.cancel()
            if let analysisID = event.id {
                latestAnalysisTask = Task { [weak self] in
                    try? await Task.sleep(nanoseconds: 12_000_000_000)
                    guard let self, self.latestAnalysisID == analysisID else { return }
                    self.latestAnalysis = nil
                    self.latestAnalysisID = nil
                }
            }
            return
        }

        if event.type == "mismatch_alert",
           let alertID = event.id,
           let quote = event.quote,
           let interpretation = event.interpretation {
            latestAnalysis = nil
            latestAnalysisID = nil
            latestAnalysisTask?.cancel()
            if var check = latestMismatchCheck, check.id == alertID {
                check.quote = quote
                check.interpretation = interpretation
                check.status = .confirmed
                latestMismatchCheck = check
            } else {
                latestMismatchCheck = MismatchCheck(
                    id: alertID,
                    transcriptID: nil,
                    quote: quote,
                    interpretation: interpretation,
                    status: .confirmed
                )
            }
            scheduleMismatchCheckDismissal(alertID)
            return
        }

        if event.type == "mismatch_review",
           let checkID = event.id,
           let outcome = event.outcome,
           var check = latestMismatchCheck,
           check.id == checkID {
            check.status = outcome == "unavailable" ? .unavailable : .unclear
            latestMismatchCheck = check
            scheduleMismatchCheckDismissal(checkID)
            return
        }

        if event.type == "llm_status", let detail = event.detail {
            errorMessage = detail
            return
        }

        guard event.type == "status", let state = event.state else { return }
        switch state {
        case "idle":
            isListening = false
            isStarting = false
            errorMessage = nil
            stateLabel = "Ready"
        case "loading_model":
            isStarting = true
            errorMessage = nil
            stateLabel = "Loading Whisper"
        case "listening":
            isListening = true
            isStarting = false
            errorMessage = nil
            stateLabel = "Listening"
        case "transcribing":
            isListening = true
            stateLabel = "Transcribing"
        case "error":
            isListening = false
            isStarting = false
            stateLabel = "Needs attention"
            errorMessage = event.detail ?? "The transcription service hit an error."
        default:
            stateLabel = state.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    private func handleCaptureFailure(_ detail: String) {
        diagnosticsTask?.cancel()
        diagnosticsTask = nil
        reportDiagnostic(
            component: "screen_capture",
            event: "capture_stopped",
            detail: detail
        )
        isListening = false
        isStarting = false
        landmarkPreview = nil
        stateLabel = "Capture stopped"
        errorMessage = detail
        microphoneCapture.stop()
        Task {
            try? await post(path: "/stop")
            ingestReceiveTask?.cancel()
            ingestReceiveTask = nil
            ingestSocket?.cancel(with: .goingAway, reason: nil)
            ingestSocket = nil
            captureMessageQueue.removeAll()
        }
    }

    private func startIngestReceiveLoop(for socket: URLSessionWebSocketTask) {
        ingestReceiveTask?.cancel()
        ingestReceiveTask = Task { [weak self, socket] in
            while !Task.isCancelled {
                do {
                    let message = try await socket.receive()
                    guard case .string(let text) = message,
                          let data = text.data(using: .utf8),
                          let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          payload["type"] as? String == "capture_error"
                    else {
                        continue
                    }
                    guard let self else { return }
                    let detail = payload["detail"] as? String
                        ?? "The local capture service rejected an audio or video packet."
                    self.reportDiagnostic(
                        component: "capture_ingest_socket",
                        event: "server_error",
                        detail: detail
                    )
                    self.errorMessage = detail
                } catch {
                    guard !Task.isCancelled,
                          let self,
                          self.ingestSocket?.taskIdentifier == socket.taskIdentifier
                    else {
                        return
                    }
                    self.reportDiagnostic(
                        component: "capture_ingest_socket",
                        event: "receive_failed",
                        detail: error.localizedDescription,
                        metadata: ["socket_close_code": socket.closeCode.rawValue]
                    )
                    self.ingestSocket = nil
                    self.captureMessageQueue.removeAll()
                    self.microphoneCapture.stop()
                    self.diagnosticsTask?.cancel()
                    self.diagnosticsTask = nil
                    self.isListening = false
                    self.isStarting = false
                    self.stateLabel = "Capture disconnected"
                    self.errorMessage = "The transcription connection closed. Restart listening to continue."
                    Task { [weak self] in
                        guard let self else { return }
                        try? await self.captureCoordinator.stop()
                        try? await self.post(path: "/stop")
                    }
                    return
                }
            }
        }
    }

    private func sendCaptureMessage(_ message: [String: Any]) {
        guard ingestSocket != nil else { return }
        guard let data = try? JSONSerialization.data(withJSONObject: message),
              let text = String(data: data, encoding: .utf8) else {
            reportDiagnostic(
                component: "capture_send_queue",
                event: "message_serialization_failed",
                detail: "A capture packet could not be encoded as JSON.",
                metadata: ["message_type": message["type"] as? String ?? "unknown"]
            )
            return
        }
        let messageType = message["type"] as? String ?? "unknown"
        capturePacketCounts[messageType, default: 0] += 1
        if messageType == "microphone_gate", message["enabled"] as? Bool == false {
            // Send a mute command ahead of unsent mic chunks so a full upload
            // queue cannot keep local speech flowing after the user mutes.
            captureMessageQueue.removeAll { $0.type == "microphone_chunk" }
        }

        if captureMessageQueue.count >= maximumCaptureQueueDepth {
            if let dropIndex = captureQueueDropIndex(for: messageType) {
                let droppedMessage = captureMessageQueue.remove(at: dropIndex)
                recordCaptureQueueDrop(
                    messageType: droppedMessage.type,
                    incomingMessageType: messageType,
                    incomingDropped: false
                )
            } else {
                recordCaptureQueueDrop(
                    messageType: messageType,
                    incomingMessageType: messageType,
                    incomingDropped: true
                )
                return
            }
        }

        let queuedMessage = QueuedCaptureMessage(type: messageType, text: text)
        if messageType == "microphone_gate" {
            captureMessageQueue.insert(queuedMessage, at: 0)
        } else {
            captureMessageQueue.append(queuedMessage)
        }
        guard captureSendTask == nil else { return }
        captureSendTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled, !self.captureMessageQueue.isEmpty {
                guard let socket = self.ingestSocket else { break }
                let item = self.captureMessageQueue.removeFirst()
                do {
                    try await socket.send(.string(item.text))
                } catch {
                    self.reportDiagnostic(
                        component: "capture_ingest_socket",
                        event: "send_failed",
                        detail: error.localizedDescription,
                        metadata: [
                            "message_type": item.type,
                            "queued_messages": self.captureMessageQueue.count,
                            "socket_close_code": socket.closeCode.rawValue
                        ]
                    )
                    self.captureMessageQueue.removeAll()
                    break
                }
            }
            self.captureSendTask = nil
        }
    }

    private func captureQueueDropIndex(for incomingMessageType: String) -> Int? {
        CaptureQueuePolicy.dropIndex(
            incomingMessageType: incomingMessageType,
            queuedMessageTypes: captureMessageQueue.map(\.type)
        )
    }

    private func recordCaptureQueueDrop(
        messageType: String,
        incomingMessageType: String,
        incomingDropped: Bool
    ) {
        captureQueueDroppedTotal += 1
        let dropped = captureQueueDroppedTotal
        guard dropped == 1 || dropped % 50 == 0 else { return }
        reportDiagnostic(
            component: "capture_send_queue",
            event: "queue_overflow",
            detail: incomingDropped
                ? "A capture packet was discarded because the send queue was full."
                : "An older capture packet was discarded to preserve newer audio.",
            metadata: [
                "dropped_total": dropped,
                "queue_depth": captureMessageQueue.count,
                "message_type": messageType,
                "incoming_message_type": incomingMessageType,
                "incoming_dropped": incomingDropped
            ]
        )
    }

    private func reportDiagnostic(
        component: String,
        event: String,
        detail: String,
        metadata: [String: Any] = [:]
    ) {
        var message: [String: Any] = [
            "component": component,
            "event": event,
            "detail": detail
        ]
        metadata.forEach { message[$0.key] = $0.value }
        guard let body = try? JSONSerialization.data(withJSONObject: message) else { return }
        Task {
            var request = URLRequest(url: serviceURL.appending(path: "/diagnostics"))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
            request.timeoutInterval = 3
            do {
                let (_, response) = try await URLSession.shared.data(for: request)
                guard let http = response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    return
                }
            } catch {
                NSLog(
                    "Subtext diagnostic could not reach the local service: %@",
                    error.localizedDescription
                )
            }
        }
    }

    private func scheduleMismatchCheckDismissal(_ checkID: String) {
        latestMismatchTask?.cancel()
        latestMismatchTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 8_000_000_000)
            guard let self, self.latestMismatchCheck?.id == checkID else { return }
            self.latestMismatchCheck = nil
        }
    }

    private func startDiagnosticsHeartbeat() {
        diagnosticsTask?.cancel()
        diagnosticsTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 10_000_000_000)
                guard let self,
                      !Task.isCancelled,
                      self.ingestSocket != nil else {
                    return
                }
                self.reportDiagnostic(
                    component: "native_capture",
                    event: "capture_heartbeat",
                    detail: "App-side packet and callback totals since capture started.",
                    metadata: [
                        "packets_created": self.capturePacketCounts,
                        "microphone_input_buffers": self.microphoneBufferCount,
                        "microphone_input_level": self.microphoneLevel,
                        "ingest_send_queue_depth": self.captureMessageQueue.count,
                        "ingest_packets_dropped": self.captureQueueDroppedTotal,
                        "overlay_socket_connected": self.isConnected
                    ]
                )
            }
        }
    }
}

private struct ServiceError: Decodable {
    let detail: String
}

private enum ClientError: LocalizedError {
    case invalidResponse
    case service(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "The local transcription service returned an invalid response."
        case .service(let message):
            message
        }
    }
}

struct OverlayView: View {
    @ObservedObject var client: TranscriptionClient

    private var impliedMeaningAlert: ImpliedMeaningSignal? {
        guard let signal = client.latestAnalysis?.impliedMeaning,
              signal.detected,
              signal.kind != "none",
              signal.confidence >= 70,
              !signal.quote.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !signal.meaning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return signal
    }

    private var showsAnalysisCard: Bool {
        client.isListening
            || client.latestMismatchCheck != nil
            || impliedMeaningAlert != nil
            || client.latestAnalysis?.noClearSignal == false
    }

    @ViewBuilder
    private var mismatchCheckCard: some View {
        if let check = client.latestMismatchCheck {
            VStack(alignment: .leading, spacing: 7) {
                HStack {
                    Label("Mismatch review", systemImage: "arrow.left.arrow.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Color.orange.opacity(0.95))
                    Spacer(minLength: 8)
                    Text(check.status.label)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(check.status.visualColor)
                        .lineLimit(1)
                }

                if !check.quote.isEmpty {
                    Text("“\(check.quote)”")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.88))
                        .lineLimit(1)
                }

                HStack(spacing: 5) {
                    mismatchStep("Words", icon: "text.quote", color: .green)
                    Capsule().fill(Color.white.opacity(0.18)).frame(width: 9, height: 1)
                    mismatchStep("Voice", icon: "waveform", color: .green)
                    Capsule().fill(Color.white.opacity(0.18)).frame(width: 9, height: 1)
                    mismatchStep(
                        "Visual cues",
                        icon: check.status == .checking ? "eye" : (check.status == .confirmed ? "checkmark" : "ellipsis"),
                        color: check.status.visualColor
                    )
                }

                let note = check.interpretation ?? check.status.detail
                if !note.isEmpty {
                    Text(note)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.white.opacity(0.64))
                        .fixedSize(horizontal: false, vertical: true)
                        .lineLimit(2)
                }
            }
            .padding(9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.orange.opacity(0.085), in: RoundedRectangle(cornerRadius: 11))
            .overlay {
                RoundedRectangle(cornerRadius: 11)
                    .stroke(Color.orange.opacity(0.20), lineWidth: 1)
            }
        }
    }

    private func mismatchStep(_ title: String, icon: String, color: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(color)
            Text(title)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.white.opacity(0.72))
                .lineLimit(1)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 5)
        .background(Color.white.opacity(0.055), in: Capsule())
    }

    private var statusColor: Color {
        if client.errorMessage != nil { return .orange }
        if client.isListening { return .green }
        if client.isConnected { return .gray }
        return .red
    }

    private var valencePanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("LIVE FACE VALENCE · ESTIMATE")
                    .font(.system(size: 9, weight: .semibold))
                    .tracking(0.8)
                    .foregroundStyle(.white.opacity(0.52))
                Spacer()
                Text("SEQUENCE MODEL")
                    .font(.system(size: 8, weight: .medium))
                    .tracking(0.7)
                    .foregroundStyle(.white.opacity(0.34))
            }

            if client.valenceEstimates.isEmpty {
                Text(valencePlaceholder)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.62))
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                ForEach(client.valenceEstimates) { estimate in
                    HStack(spacing: 8) {
                        Text(client.valencePersonNames[estimate.trackID] ?? estimate.personLabel)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.white.opacity(0.70))
                            .lineLimit(1)
                            .truncationMode(.tail)
                            .frame(width: 76, alignment: .leading)
                        if let score = estimate.score {
                            valenceMeter(score: score)
                            Text(String(format: "%+.2f", score))
                                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.90))
                                .frame(width: 42, alignment: .trailing)
                        } else {
                            Text("Gathering frames \(estimate.framesSeen)/10…")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.white.opacity(0.55))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                HStack {
                    Text("negative")
                    Spacer()
                    Text("positive")
                }
                .font(.system(size: 8, weight: .medium))
                .foregroundStyle(.white.opacity(0.38))
                .padding(.leading, 84)
            }
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.045), in: RoundedRectangle(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.white.opacity(0.07), lineWidth: 1)
        }
    }

    private var valencePlaceholder: String {
        switch client.valenceModelState {
        case "loading":
            client.valenceModelDetail ?? "Loading the local sequence model…"
        case "error":
            client.valenceModelDetail ?? "The face model could not start."
        case "ready":
            "No face is currently visible."
        default:
            client.isListening ? "Waiting for face samples…" : "Start listening to see a live estimate."
        }
    }

    private func valenceMeter(score: Double) -> some View {
        GeometryReader { geometry in
            let markerX = CGFloat(min(max((score + 1) / 2, 0), 1)) * geometry.size.width
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [.red.opacity(0.85), .orange.opacity(0.8), .gray.opacity(0.65), .green.opacity(0.9)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(height: 6)
                Rectangle()
                    .fill(Color.white.opacity(0.42))
                    .frame(width: 1, height: 9)
                    .position(x: geometry.size.width / 2, y: 4.5)
                Circle()
                    .fill(.white)
                    .overlay(Circle().stroke(Color.black.opacity(0.35), lineWidth: 1))
                    .frame(width: 9, height: 9)
                    .position(x: markerX, y: 4.5)
            }
            .animation(.easeOut(duration: 0.18), value: score)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 9)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "waveform")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(Color.white.opacity(0.13), in: RoundedRectangle(cornerRadius: 11))

                VStack(alignment: .leading, spacing: 2) {
                    Text("Subtext")
                        .font(.system(size: 16, weight: .semibold))
                    HStack(spacing: 6) {
                        Circle().fill(statusColor).frame(width: 6, height: 6)
                        Text(client.stateLabel)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.white.opacity(0.67))
                    }
                }

                Spacer()

                Button(action: client.clearTranscript) {
                    Image(systemName: "trash")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.65))
                        .frame(width: 28, height: 28)
                        .background(Color.white.opacity(0.08), in: Circle())
                }
                .buttonStyle(.plain)
                .help("Clear transcript")

                Button(action: { NSApp.terminate(nil) }) {
                    Image(systemName: "xmark")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.65))
                        .frame(width: 28, height: 28)
                        .background(Color.white.opacity(0.08), in: Circle())
                }
                .buttonStyle(.plain)
                .help("Quit Subtext")
                .accessibilityLabel("Quit Subtext")
            }

            Divider().overlay(Color.white.opacity(0.10)).padding(.vertical, 11)

            HStack(spacing: 8) {
                Image(systemName: "rectangle.inset.filled")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.62))
                Picker("Call window", selection: $client.selectedWindowID) {
                    Text("Choose call window").tag(nil as UInt32?)
                    ForEach(client.windows) { window in
                        Text(window.title).tag(Optional(window.id))
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity, alignment: .leading)
                .disabled(client.isListening || client.isStarting)

                Button {
                    Task { await client.refreshWindows() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.white.opacity(0.72))
                        .frame(width: 26, height: 26)
                        .background(Color.white.opacity(0.08), in: Circle())
                }
                .buttonStyle(.plain)
                .help("Refresh call windows")
                .disabled(client.isListening || client.isStarting)
            }
            .padding(.bottom, 5)

            Text(client.selectedWindowID == nil
                 ? "Choose a call window to include the other person's audio; your microphone can start now."
                 : (client.geminiConfigured
                    ? "Raw audio and video stay here; transcript and measured cues go to Gemini."
                    : "Audio, video, and transcription stay here. Add a Gemini key to enable interpretation."))
                .font(.system(size: 10))
                .foregroundStyle(.white.opacity(0.46))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.bottom, 9)

            microphoneMeter
                .padding(.bottom, 4)

            HStack {
                Text("LIVE WINDOW")
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(1.2)
                    .foregroundStyle(.white.opacity(0.48))
                Spacer()
                Text("FACE LANDMARKS · ON DEVICE")
                    .font(.system(size: 9, weight: .medium))
                    .tracking(0.5)
                    .foregroundStyle(.white.opacity(0.38))
            }
            .padding(.bottom, 6)

            LandmarkPreviewView(frame: client.landmarkPreview, isListening: client.isListening)
                .frame(height: 148)
                .padding(.bottom, 9)

            valencePanel
                .padding(.bottom, 10)

            HStack {
                Text("LIVE TRANSCRIPT")
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(1.2)
                    .foregroundStyle(.white.opacity(0.48))
                Spacer()
                Text(client.geminiConfigured ? "LOCAL CAPTURE · GEMINI" : "LOCAL CAPTURE")
                    .font(.system(size: 9, weight: .medium))
                    .tracking(0.8)
                    .foregroundStyle(.white.opacity(0.38))
            }
            .padding(.bottom, 8)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 13) {
                        if client.transcript.isEmpty {
                            Text(client.isListening ? "Listening for speech…" : "Your transcript will appear here.")
                                .font(.system(size: 13))
                                .foregroundStyle(.white.opacity(0.50))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.top, 9)
                        } else {
                            ForEach(client.transcript) { entry in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack(spacing: 6) {
                                        Text(entry.speakerID == "self" ? "You" : "Other person")
                                            .font(.system(size: 10, weight: .semibold))
                                            .foregroundStyle(.white.opacity(0.58))
                                        Text(entry.timestamp.formatted(date: .omitted, time: .shortened))
                                        .font(.system(size: 10, weight: .medium))
                                        .foregroundStyle(.white.opacity(0.42))
                                    }
                                    Text(entry.text)
                                        .font(.system(size: 14))
                                        .foregroundStyle(.white.opacity(0.94))
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                                .id(entry.id)
                            }
                        }
                        Color.clear.frame(height: 1).id("transcript-bottom")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .scrollIndicators(.hidden)
                .onChange(of: client.transcript.count) { _ in
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo("transcript-bottom", anchor: .bottom)
                    }
                }
            }
            .frame(height: showsAnalysisCard ? 130 : 165)

            if client.latestMismatchCheck != nil {
                mismatchCheckCard
                    .padding(.top, 7)
            } else if let signal = impliedMeaningAlert {
                VStack(alignment: .leading, spacing: 5) {
                    Label("Possible implied meaning", systemImage: "text.quote")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Color.orange.opacity(0.95))
                    Text("“\(signal.quote)”")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.95))
                        .fixedSize(horizontal: false, vertical: true)
                        .lineLimit(1)
                    Text(signal.meaning)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.88))
                        .fixedSize(horizontal: false, vertical: true)
                        .lineLimit(2)
                    HStack {
                        Text(signal.kind.capitalized)
                        Spacer()
                        Text("Model confidence \(signal.confidence)/100")
                    }
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.white.opacity(0.60))
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.11), in: RoundedRectangle(cornerRadius: 12))
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.orange.opacity(0.25), lineWidth: 1)
                }
                .padding(.top, 7)
            } else if let analysis = client.latestAnalysis, !analysis.noClearSignal {
                VStack(alignment: .leading, spacing: 4) {
                    Label("Possible moment", systemImage: "quote.bubble")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Color(red: 0.76, green: 0.82, blue: 1.0))
                    Text(analysis.hypotheses.first?.interpretation ?? analysis.summary)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.92))
                        .fixedSize(horizontal: false, vertical: true)
                        .lineLimit(2)
                    if let checkIn = analysis.hypotheses.first?.possibleCheckIn {
                        Text(checkIn)
                            .font(.system(size: 11))
                            .foregroundStyle(.white.opacity(0.63))
                            .fixedSize(horizontal: false, vertical: true)
                            .lineLimit(2)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
                .padding(.top, 7)
            } else if client.isListening {
                VStack(alignment: .leading, spacing: 7) {
                    HStack {
                        Label("Mismatch analysis", systemImage: "arrow.left.arrow.right")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.66))
                        Spacer()
                        Circle().fill(Color.green.opacity(0.85)).frame(width: 5, height: 5)
                        Text("LISTENING")
                            .font(.system(size: 8, weight: .semibold))
                            .tracking(0.7)
                            .foregroundStyle(.white.opacity(0.48))
                    }
                    HStack(spacing: 5) {
                        mismatchStep("Words + voice", icon: "waveform", color: .white.opacity(0.62))
                        Capsule().fill(Color.white.opacity(0.18)).frame(width: 9, height: 1)
                        mismatchStep("Visual review", icon: "eye", color: .white.opacity(0.42))
                    }
                    Text("Local cues can trigger a visual-context review.")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.white.opacity(0.48))
                }
                .padding(9)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.white.opacity(0.045), in: RoundedRectangle(cornerRadius: 11))
                .overlay {
                    RoundedRectangle(cornerRadius: 11)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                }
                .padding(.top, 7)
            }

            if let error = client.errorMessage {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundStyle(Color.orange.opacity(0.95))
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 8)
            }

            Spacer(minLength: 12)

            Button {
                Task {
                    if client.isListening {
                        await client.stop()
                    } else {
                        await client.start()
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: client.isListening ? "stop.fill" : "mic.fill")
                        .font(.system(size: 12, weight: .semibold))
                    Text(client.isStarting ? "Starting…" : (client.isListening ? "Stop listening" : "Start listening"))
                        .font(.system(size: 13, weight: .semibold))
                    Spacer()
                    if !client.isConnected {
                        Text("Connecting…")
                            .font(.system(size: 10, weight: .medium))
                            .opacity(0.68)
                    }
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .frame(height: 42)
                .background(client.isListening ? Color.white.opacity(0.16) : Color(red: 0.31, green: 0.47, blue: 0.94), in: RoundedRectangle(cornerRadius: 13))
            }
            .buttonStyle(.plain)
            .disabled(!client.isConnected || client.isStarting)
            .opacity(client.isConnected && !client.isStarting ? 1 : 0.55)
        }
        .foregroundStyle(.white)
        .padding(20)
        .frame(width: 390, height: 720)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 23))
        .overlay {
            RoundedRectangle(cornerRadius: 23)
                .stroke(Color.white.opacity(0.13), lineWidth: 1)
        }
        .onAppear {
            Task { await client.refreshWindows() }
        }
    }

    private var microphoneMeter: some View {
        let decibels = 20 * log10(max(client.microphoneLevel, 0.00001))
        let fill = min(1, max(0, (decibels + 60) / 60))
        let status: String
        if !client.isListening {
            status = "Not listening"
        } else if client.microphoneMuted {
            status = "Muted · others continue"
        } else if !client.microphoneBufferReceived {
            status = "No mic samples"
        } else if client.microphoneLevel > 0.001 {
            status = "Input detected"
        } else {
            status = "Mic active · quiet"
        }
        return HStack(spacing: 8) {
            Image(systemName: client.microphoneMuted ? "mic.slash.fill" : "mic.fill")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white.opacity(0.58))
            Text("MICROPHONE INPUT")
                .font(.system(size: 8, weight: .semibold))
                .tracking(0.8)
                .foregroundStyle(.white.opacity(0.48))
            Spacer()
            Text(status)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.white.opacity(0.48))
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.11))
                    Capsule()
                        .fill(fill > 0.72 ? Color.orange : Color.green)
                        .frame(width: geometry.size.width * fill)
                }
            }
            .frame(width: 52, height: 5)
            Button(action: client.toggleMicrophoneMuted) {
                Image(systemName: client.microphoneMuted ? "mic.slash.fill" : "mic.fill")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.white.opacity(client.isListening ? 0.82 : 0.38))
                    .frame(width: 22, height: 18)
                    .background(Color.white.opacity(client.microphoneMuted ? 0.16 : 0.07), in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(!client.isListening)
            .help(client.microphoneMuted
                ? "Resume local microphone transcription. Call-window audio is still transcribed."
                : "Mute local microphone transcription. Mirror your call's mute state here; call-window audio will continue.")
        }
        .frame(height: 14)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Microphone level")
    }
}

private struct LandmarkPreviewView: View {
    let frame: LandmarkPreviewFrame?
    let isListening: Bool

    var body: some View {
        Group {
            if let frame,
               let image = NSImage(data: frame.jpegData),
               let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                GeometryReader { geometry in
                    ZStack {
                        Color.black
                        Image(decorative: cgImage, scale: 1)
                            .resizable()
                            .scaledToFit()

                        Canvas { context, size in
                            let imageRect = Self.aspectFitRect(
                                imageSize: CGSize(width: CGFloat(cgImage.width), height: CGFloat(cgImage.height)),
                                in: size
                            )

                            for face in frame.faces {
                                for point in face.points where point.count == 2 {
                                    let x = imageRect.minX + CGFloat(point[0]) * imageRect.width
                                    let y = imageRect.minY + (1 - CGFloat(point[1])) * imageRect.height
                                    let dot = CGRect(x: x - 2, y: y - 2, width: 4, height: 4)
                                    context.fill(Path(ellipseIn: dot), with: .color(.cyan))
                                }
                            }
                        }

                        if frame.faces.isEmpty {
                            Text("No face detected")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.white.opacity(0.75))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(.black.opacity(0.55), in: Capsule())
                        }
                    }
                    .overlay(alignment: .topLeading) {
                        Text("LIVE · 5 FPS")
                            .font(.system(size: 9, weight: .semibold))
                            .tracking(0.5)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 4)
                            .background(.black.opacity(0.55), in: Capsule())
                            .padding(7)
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay {
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.white.opacity(0.12), lineWidth: 1)
                    }
                    .frame(width: geometry.size.width, height: geometry.size.height)
                }
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.black.opacity(0.28))
                    Label(
                        isListening ? "Waiting for call video…" : "Start listening to preview the call window",
                        systemImage: "viewfinder"
                    )
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.52))
                    .padding(.horizontal, 12)
                    .multilineTextAlignment(.center)
                }
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.10), lineWidth: 1)
                }
            }
        }
    }

    private static func aspectFitRect(imageSize: CGSize, in containerSize: CGSize) -> CGRect {
        let scale = min(containerSize.width / imageSize.width, containerSize.height / imageSize.height)
        let fittedSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(
            x: (containerSize.width - fittedSize.width) / 2,
            y: (containerSize.height - fittedSize.height) / 2,
            width: fittedSize.width,
            height: fittedSize.height
        )
    }

}

private struct SpeakerTileGlow: View {
    let rect: CGRect
    let wavePhase: CGFloat
    let flowDegrees: Double
    let pulse: Double
    let valenceScore: Double?

    private var cornerRadius: CGFloat { min(28, min(rect.width, rect.height) * 0.055) }
    private var valencePalette: (tint: Color, highlight: Color) {
        let negative = (red: 1.0, green: 0.16, blue: 0.31)
        let neutral = (red: 0.48, green: 0.55, blue: 0.66)
        let positive = (red: 0.08, green: 0.96, blue: 0.49)
        guard let valenceScore, valenceScore.isFinite else {
            return palette(for: neutral)
        }

        let score = min(max(valenceScore, -1), 1)
        let start = score < 0 ? negative : neutral
        let end = score < 0 ? neutral : positive
        let amount = score < 0 ? score + 1 : score
        let red = start.red + (end.red - start.red) * amount
        let green = start.green + (end.green - start.green) * amount
        let blue = start.blue + (end.blue - start.blue) * amount
        return palette(for: (red: red, green: green, blue: blue))
    }

    private func palette(for color: (red: Double, green: Double, blue: Double)) -> (tint: Color, highlight: Color) {
        (
            Color(red: color.red, green: color.green, blue: color.blue),
            Color(
                red: color.red + (1 - color.red) * 0.42,
                green: color.green + (1 - color.green) * 0.42,
                blue: color.blue + (1 - color.blue) * 0.42
            )
        )
    }

    var body: some View {
        ZStack {
            // The broad halo and tighter filament follow the tile while moving outside its edge.
            LiquidAuraRim(cornerRadius: cornerRadius, phase: wavePhase, amplitude: 5.5, outset: 8)
                .stroke(
                    AngularGradient(
                        stops: [
                            .init(color: valencePalette.tint.opacity(0.88), location: 0),
                            .init(color: valencePalette.tint.opacity(0.62), location: 0.24),
                            .init(color: Color.white.opacity(0.95), location: 0.48),
                            .init(color: valencePalette.highlight.opacity(0.82), location: 0.70),
                            .init(color: valencePalette.tint.opacity(0.88), location: 1)
                        ],
                        center: .center,
                        startAngle: .degrees(flowDegrees),
                        endAngle: .degrees(flowDegrees + 360)
                    ),
                    style: StrokeStyle(lineWidth: 11, lineCap: .round, lineJoin: .round)
                )
                .blur(radius: 10 + 3 * pulse)
                .opacity(0.5 + 0.18 * pulse)

            LiquidAuraRim(cornerRadius: cornerRadius, phase: wavePhase * 0.78 + 1.1, amplitude: 2.8, outset: 5.5)
                .stroke(
                    AngularGradient(
                        colors: [valencePalette.tint.opacity(0.92), valencePalette.highlight.opacity(0.86), .white, valencePalette.highlight.opacity(0.9), valencePalette.tint.opacity(0.92)],
                        center: .center,
                        startAngle: .degrees(flowDegrees - 35),
                        endAngle: .degrees(flowDegrees + 325)
                    ),
                    style: StrokeStyle(lineWidth: 3.4, lineCap: .round, lineJoin: .round)
                )
                .blur(radius: 1.5 + pulse)
                .shadow(color: valencePalette.tint.opacity(0.78 + 0.16 * pulse), radius: 7 + 4 * pulse)
                .opacity(0.88)
        }
        .frame(width: rect.width, height: rect.height)
        .position(x: rect.midX, y: rect.midY)
        .animation(.easeInOut(duration: 0.45), value: rect)
        .animation(.easeInOut(duration: 0.85), value: valenceScore)
    }
}

/// A softly irregular rounded rectangle. Its moving contour follows the tile's
/// rounded silhouette while sitting just outside the video panel.
private struct LiquidAuraRim: Shape {
    var cornerRadius: CGFloat
    var phase: CGFloat
    var amplitude: CGFloat
    var outset: CGFloat = 0

    func path(in rect: CGRect) -> Path {
        let bounds = rect.insetBy(dx: -outset, dy: -outset)
        let width = bounds.width
        let height = bounds.height
        guard width > 0, height > 0 else { return Path() }

        let radius = min(max(0, cornerRadius + outset), min(width / 2, height / 2))
        let horizontalLength = max(0, width - 2 * radius)
        let verticalLength = max(0, height - 2 * radius)
        let perimeter = 2 * (horizontalLength + verticalLength) + 2 * .pi * radius
        guard perimeter > 0 else { return Path() }

        var path = Path()
        var distance: CGFloat = 0

        func displacedPoint(_ point: CGPoint, normal: CGPoint, at distance: CGFloat) -> CGPoint {
            let progress = distance / perimeter
            let primaryWave = sin(progress * 2 * .pi * 4 - phase)
            let secondaryWave = sin(progress * 2 * .pi * 7 + phase * 0.67)
            let ripple = amplitude * (primaryWave * 0.72 + secondaryWave * 0.28)
            return CGPoint(x: point.x + normal.x * ripple, y: point.y + normal.y * ripple)
        }

        func appendSegment(length: CGFloat, pointAt: (CGFloat) -> CGPoint, normalAt: (CGFloat) -> CGPoint) {
            let steps = max(2, Int(ceil(length / 5)))
            for step in 1...steps {
                let fraction = CGFloat(step) / CGFloat(steps)
                let point = pointAt(fraction)
                let normal = normalAt(fraction)
                path.addLine(to: displacedPoint(point, normal: normal, at: distance + length * fraction))
            }
            distance += length
        }

        let start = CGPoint(x: bounds.minX + radius, y: bounds.minY)
        path.move(to: displacedPoint(start, normal: CGPoint(x: 0, y: -1), at: 0))

        appendSegment(
            length: horizontalLength,
            pointAt: { t in CGPoint(x: bounds.minX + radius + horizontalLength * t, y: bounds.minY) },
            normalAt: { _ in CGPoint(x: 0, y: -1) }
        )
        appendSegment(
            length: .pi * radius / 2,
            pointAt: { t in
                let angle = -.pi / 2 + .pi / 2 * t
                return CGPoint(x: bounds.maxX - radius + cos(angle) * radius, y: bounds.minY + radius + sin(angle) * radius)
            },
            normalAt: { t in
                let angle = -.pi / 2 + .pi / 2 * t
                return CGPoint(x: cos(angle), y: sin(angle))
            }
        )
        appendSegment(
            length: verticalLength,
            pointAt: { t in CGPoint(x: bounds.maxX, y: bounds.minY + radius + verticalLength * t) },
            normalAt: { _ in CGPoint(x: 1, y: 0) }
        )
        appendSegment(
            length: .pi * radius / 2,
            pointAt: { t in
                let angle = .pi / 2 * t
                return CGPoint(x: bounds.maxX - radius + cos(angle) * radius, y: bounds.maxY - radius + sin(angle) * radius)
            },
            normalAt: { t in
                let angle = .pi / 2 * t
                return CGPoint(x: cos(angle), y: sin(angle))
            }
        )
        appendSegment(
            length: horizontalLength,
            pointAt: { t in CGPoint(x: bounds.maxX - radius - horizontalLength * t, y: bounds.maxY) },
            normalAt: { _ in CGPoint(x: 0, y: 1) }
        )
        appendSegment(
            length: .pi * radius / 2,
            pointAt: { t in
                let angle = .pi / 2 + .pi / 2 * t
                return CGPoint(x: bounds.minX + radius + cos(angle) * radius, y: bounds.maxY - radius + sin(angle) * radius)
            },
            normalAt: { t in
                let angle = .pi / 2 + .pi / 2 * t
                return CGPoint(x: cos(angle), y: sin(angle))
            }
        )
        appendSegment(
            length: verticalLength,
            pointAt: { t in CGPoint(x: bounds.minX, y: bounds.maxY - radius - verticalLength * t) },
            normalAt: { _ in CGPoint(x: -1, y: 0) }
        )
        appendSegment(
            length: .pi * radius / 2,
            pointAt: { t in
                let angle = .pi + .pi / 2 * t
                return CGPoint(x: bounds.minX + radius + cos(angle) * radius, y: bounds.minY + radius + sin(angle) * radius)
            },
            normalAt: { t in
                let angle = .pi + .pi / 2 * t
                return CGPoint(x: cos(angle), y: sin(angle))
            }
        )
        path.closeSubpath()
        return path
    }
}

private struct SpeakerTileOverlayView: View {
    @ObservedObject var client: TranscriptionClient

    var body: some View {
        GeometryReader { geometry in
            if let frame = client.landmarkPreview,
               let tileBox = frame.speakerTileBox,
               let rect = tileRect(tileBox, in: geometry.size) {
                let speakerValence = frame.speakerTrackID.flatMap { trackID in
                    client.valenceEstimates.first(where: { $0.trackID == trackID })?.score
                }
                TimelineView(.animation(
                    minimumInterval: 1.0 / 60.0,
                    paused: !client.isListening
                )) { timeline in
                    let elapsed = timeline.date.timeIntervalSinceReferenceDate
                    let phase = CGFloat(elapsed * 2 * .pi / 3.4)
                    let flowDegrees = (elapsed * 42).truncatingRemainder(dividingBy: 360)
                    SpeakerTileGlow(
                        rect: rect,
                        wavePhase: phase,
                        flowDegrees: flowDegrees,
                        pulse: (sin(elapsed * 2 * .pi / 3.6) + 1) / 2,
                        valenceScore: speakerValence
                    )
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.clear)
        .allowsHitTesting(false)
    }

    private func tileRect(_ box: [Double], in size: CGSize) -> CGRect? {
        guard box.count == 4 else { return nil }
        let x = CGFloat(box[0])
        let y = CGFloat(box[1])
        let width = CGFloat(box[2])
        let height = CGFloat(box[3])
        guard x.isFinite, y.isFinite, width.isFinite, height.isFinite,
              width > 0, height > 0 else {
            return nil
        }
        return CGRect(
            x: x * size.width,
            y: (1 - y - height) * size.height,
            width: width * size.width,
            height: height * size.height
        )
    }
}

final class MicrophoneCaptureCoordinator {
    private let sampleRate = 16_000.0
    private var engine = AVAudioEngine()
    private let muteStateLock = NSLock()
    private var muted = false
    private var inputNode: AVAudioInputNode?
    private var inputTapInstalled = false
    private var converter: AVAudioConverter?
    private var outputFormat: AVAudioFormat?
    private var engineConfigurationObserver: NSObjectProtocol?
    private let defaultInputDeviceListenerQueue = DispatchQueue(
        label: "org.subtext.capture.default-microphone"
    )
    private var defaultInputDeviceListener: AudioObjectPropertyListenerBlock?
    private var inputWatchdogTask: Task<Void, Never>?
    private var hasReportedUnsupportedInputFormat = false
    private let inputStateLock = NSLock()
    private var lastInputBufferAt = Date.distantPast
    private var hasReportedInputStall = false
    private var lastRecoveryAttemptAt = Date.distantPast
    private var isReconfiguring = false
    private(set) var isCapturing = false
    var messageHandler: (([String: Any]) -> Void)?
    var levelHandler: ((Double) -> Void)?
    var bufferHandler: (() -> Void)?
    var inputStalledHandler: (() -> Void)?
    var diagnosticHandler: ((String, String) -> Void)?

    init() {
        engineConfigurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let changedEngine = notification.object as? AVAudioEngine else { return }
            // The engine posts this from an internal queue and has already
            // stopped/uninitialized its I/O unit. Rebuild after the notification
            // callback returns so teardown doesn't run inside that callback.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self, weak changedEngine] in
                guard let self, let changedEngine, self.engine === changedEngine else { return }
                self.recoverInput(reason: "audio engine configuration changed")
            }
        }

        var address = Self.defaultInputDevicePropertyAddress
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            // The default device can change without changing sample rate or
            // channel count, so AVAudioEngine may keep delivering silent
            // buffers without posting its configuration-change notification.
            // Rebuild after Core Audio finishes switching the route.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
                self?.recoverInput(reason: "macOS default microphone changed")
            }
        }
        let listenerStatus = AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            defaultInputDeviceListenerQueue,
            listener
        )
        if listenerStatus == noErr {
            defaultInputDeviceListener = listener
        } else {
            NSLog("Subtext could not monitor the default microphone route: %d", listenerStatus)
        }
    }

    deinit {
        inputWatchdogTask?.cancel()
        if let engineConfigurationObserver {
            NotificationCenter.default.removeObserver(engineConfigurationObserver)
        }
        if let defaultInputDeviceListener {
            var address = Self.defaultInputDevicePropertyAddress
            AudioObjectRemovePropertyListenerBlock(
                AudioObjectID(kAudioObjectSystemObject),
                &address,
                defaultInputDeviceListenerQueue,
                defaultInputDeviceListener
            )
        }
    }

    func start() async throws {
        guard !isCapturing else { return }
        guard await microphonePermissionGranted() else {
            throw MicrophoneCaptureError.permissionDenied
        }

        do {
            try configureInputTap()
            engine.prepare()
            try engine.start()
            isCapturing = true
            markInputBufferClockStarted()
            startInputWatchdog()
        } catch {
            replaceAudioEngine()
            throw error
        }
    }

    private func configureInputTap() throws {
        guard !inputTapInstalled else {
            throw MicrophoneCaptureError.inputTapAlreadyInstalled
        }
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            throw MicrophoneCaptureError.inputUnavailable
        }
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw MicrophoneCaptureError.outputFormatUnavailable
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            diagnosticHandler?(
                "audio_converter_unavailable",
                "Could not convert input format \(inputFormat) to mono 16 kHz float32."
            )
            throw MicrophoneCaptureError.converterUnavailable(inputFormat.description)
        }
        // Keep the backend contract mono while mapping one measured input
        // channel directly. A multi-channel output format can fail to initialize
        // for devices whose input layout has more than two channels.
        converter.downmix = false
        converter.channelMap = [NSNumber(value: 0)]

        self.inputNode = inputNode
        self.converter = converter
        self.outputFormat = outputFormat
        diagnosticHandler?(
            "input_format",
            "Using macOS default input \(Self.defaultInputDeviceDescription()): "
                + "\(Int(inputFormat.sampleRate)) Hz, "
                + "\(inputFormat.channelCount) channel(s), \(inputFormat.commonFormat); "
                + "routing the strongest input channel to mono 16 kHz."
        )
        inputNode.installTap(onBus: 0, bufferSize: 4_096, format: inputFormat) { [weak self] buffer, _ in
            self?.convertAndSend(buffer)
        }
        inputTapInstalled = true
    }

    func stop() {
        guard isCapturing else { return }
        isCapturing = false
        inputWatchdogTask?.cancel()
        inputWatchdogTask = nil
        replaceAudioEngine()
    }

    func setMuted(_ muted: Bool) {
        muteStateLock.lock()
        self.muted = muted
        muteStateLock.unlock()
    }

    private func isMuted() -> Bool {
        muteStateLock.lock()
        defer { muteStateLock.unlock() }
        return muted
    }

    private func clearInputConfiguration() {
        inputNode = nil
        converter = nil
        outputFormat = nil
    }

    private func removeInputTap() {
        inputNode?.removeTap(onBus: 0)
        inputTapInstalled = false
    }

    /// Retire the old audio graph before retrying. AVAudioEngine can retain a
    /// tap across stop/reconfiguration on macOS, and installTap raises an
    /// Objective-C exception (which Swift cannot catch) if that tap is still
    /// present. A fresh engine gives every recovery attempt an empty graph.
    private func replaceAudioEngine() {
        let previousEngine = engine
        previousEngine.stop()
        removeInputTap()
        clearInputConfiguration()
        engine = AVAudioEngine()
    }

    private func markInputBufferClockStarted() {
        inputStateLock.lock()
        lastInputBufferAt = Date()
        hasReportedInputStall = false
        inputStateLock.unlock()
    }

    private func markInputBufferReceived() {
        inputStateLock.lock()
        lastInputBufferAt = Date()
        let wasStalled = hasReportedInputStall
        hasReportedInputStall = false
        inputStateLock.unlock()
        if wasStalled {
            DispatchQueue.main.async { [weak self] in
                self?.diagnosticHandler?("input_resumed", "Microphone buffers resumed after an input stall.")
            }
        }
    }

    private func inputBufferIsStalled() -> Bool {
        inputStateLock.lock()
        let stalled = Date().timeIntervalSince(lastInputBufferAt) > 3
        let shouldReport = stalled && !hasReportedInputStall
        if shouldReport {
            hasReportedInputStall = true
        }
        inputStateLock.unlock()
        if shouldReport {
            DispatchQueue.main.async { [weak self] in
                self?.inputStalledHandler?()
                self?.diagnosticHandler?("input_stalled", "No microphone input buffers arrived for more than 3 seconds.")
            }
        }
        return stalled
    }

    private func startInputWatchdog() {
        inputWatchdogTask?.cancel()
        inputWatchdogTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let self, self.isCapturing else { return }
                guard self.inputBufferIsStalled() else { continue }
                self.recoverInput(reason: "microphone input callback stalled")
            }
        }
    }

    private func recoverInput(reason: String) {
        guard isCapturing, !isReconfiguring else { return }
        guard Date().timeIntervalSince(lastRecoveryAttemptAt) >= 3 else { return }
        lastRecoveryAttemptAt = Date()
        isReconfiguring = true
        defer { isReconfiguring = false }

        diagnosticHandler?("input_recovery_started", reason)
        replaceAudioEngine()

        do {
            try configureInputTap()
            engine.prepare()
            try engine.start()
            markInputBufferClockStarted()
            diagnosticHandler?(
                "input_recovery_succeeded",
                "Microphone engine restarted after \(reason)."
            )
        } catch {
            replaceAudioEngine()
            diagnosticHandler?(
                "input_recovery_failed",
                "Microphone engine restart failed after \(reason): \(error.localizedDescription)"
            )
        }
    }

    private func microphonePermissionGranted() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .audio)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private static var defaultInputDevicePropertyAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
    }

    private static func defaultInputDeviceDescription() -> String {
        var deviceID = AudioDeviceID(kAudioObjectUnknown)
        var address = defaultInputDevicePropertyAddress
        var dataSize = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &dataSize,
            &deviceID
        )
        guard status == noErr, deviceID != kAudioObjectUnknown else {
            return "unavailable (Core Audio status \(status))"
        }

        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var unmanagedName: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(
            deviceID,
            &nameAddress,
            0,
            nil,
            &nameSize,
            &unmanagedName
        ) == noErr, let unmanagedName else {
            return "device \(deviceID)"
        }
        return "\(unmanagedName.takeUnretainedValue() as String) (device \(deviceID))"
    }

    private func convertAndSend(_ inputBuffer: AVAudioPCMBuffer) {
        guard inputBuffer.frameLength > 0 else { return }
        markInputBufferReceived()
        bufferHandler?()
        guard let converter, let outputFormat else { return }

        // Measure the loudest input channel. Some audio devices expose Int16
        // or Int32 buffers, and averaging a microphone array can cancel its
        // channels when they arrive with different phase.
        let inputLevels = AudioInputLevelMeter.channelRMSLevels(in: inputBuffer) ?? []
        let strongestInputChannel = inputLevels.enumerated().max { $0.element < $1.element }?.offset ?? 0
        if let level = inputLevels.max() {
            levelHandler?(level)
        } else {
            levelHandler?(0)
            if !hasReportedUnsupportedInputFormat {
                hasReportedUnsupportedInputFormat = true
                diagnosticHandler?(
                    "unsupported_input_format",
                    "Microphone buffers arrived in an unsupported PCM format: "
                        + "\(inputBuffer.format.commonFormat)."
                )
            }
        }

        // The call-window stream remains active while local microphone
        // transcription is muted.
        guard !isMuted() else { return }
        converter.channelMap = [NSNumber(value: strongestInputChannel)]

        let ratio = outputFormat.sampleRate / inputBuffer.format.sampleRate
        let outputCapacity = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio) + 32)
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: outputCapacity
        ) else {
            return
        }

        var suppliedInput = false
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, inputStatus in
            guard !suppliedInput else {
                inputStatus.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            inputStatus.pointee = .haveData
            return inputBuffer
        }
        guard status != .error,
              outputBuffer.frameLength > 0,
              let channels = outputBuffer.floatChannelData
        else {
            if status == .error {
                diagnosticHandler?(
                    "audio_conversion_failed",
                    conversionError?.localizedDescription ?? "The microphone audio converter failed."
                )
            }
            return
        }

        let frameCount = Int(outputBuffer.frameLength)
        let samples = channels[0]

        let byteCount = frameCount * MemoryLayout<Float>.size
        let pcm = Data(bytes: samples, count: byteCount)
        let message: [String: Any] = [
            "type": "microphone_chunk",
            "timestamp": Date().timeIntervalSince1970,
            "sample_rate": Int(sampleRate),
            "pcm_format": "float32",
            "samples": pcm.base64EncodedString()
        ]
        messageHandler?(message)
    }

}

private enum MicrophoneCaptureError: LocalizedError {
    case permissionDenied
    case inputUnavailable
    case inputTapAlreadyInstalled
    case outputFormatUnavailable
    case converterUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            "Subtext needs microphone access. Allow it in System Settings → Privacy & Security → Microphone, then restart Subtext."
        case .inputUnavailable:
            "No microphone input is available. Choose an input in System Settings → Sound → Input."
        case .inputTapAlreadyInstalled:
            "Subtext could not restart microphone capture safely. Stop listening and start again."
        case .outputFormatUnavailable:
            "Subtext could not prepare mono 16 kHz microphone audio."
        case .converterUnavailable(let inputFormat):
            "Subtext could not convert the selected microphone format (\(inputFormat)) to mono 16 kHz audio."
        }
    }
}

struct ShareableWindow: Identifiable {
    let id: UInt32
    let title: String
    fileprivate let nativeWindow: SCWindow
}

/// Captures one selected call window. Audio, numeric cues, and small face crops are
/// sent only to the local Python service for transcription and temporal inference.
@MainActor
final class ScreenCaptureCoordinator: NSObject, ObservableObject, SCStreamOutput, SCStreamDelegate {
    private struct FaceTrackState {
        var boundingBox: CGRect
        var lastSeen: TimeInterval
    }

    private struct SpeakerTileState {
        let trackID: String
        let bounds: CGRect
    }

    @Published private(set) var windows: [ShareableWindow] = []
    @Published private(set) var isCapturing = false
    @Published private(set) var errorMessage: String?

    private var stream: SCStream?
    // ScreenCaptureKit emits many short call-audio callbacks. Batch them off
    // the main actor before JSON/base64 encoding to keep the websocket sender
    // from falling behind during meetings.
    nonisolated private let audioBatcher = AudioChunkBatcher(targetDuration: 0.1)
    private var faceTracks: [String: FaceTrackState] = [:]
    private var speakerTileState: SpeakerTileState?
    private var stableVisibleFaceCount: Int?
    private var pendingVisibleFaceCount: Int?
    private var pendingVisibleFaceCountFrames = 0
    private var pendingSpeakerTileBounds: [CGRect] = []
    private var collectingSpeakerTileLayout = true
    private var pendingSpeakerTileMisses = 0
    private var nextFaceTrackNumber = 1
    private var activeWindow: SCWindow?
    private var activeWindowScreenFrame: CGRect?
    private var activeWindowProcessID: pid_t?
    var messageHandler: (([String: Any]) -> Void)?
    var personNameUpdateHandler: ((String, String) -> Void)?
    var previewHandler: ((LandmarkPreviewFrame?) -> Void)?
    var windowFrameHandler: ((CGRect?, pid_t?, UInt32?) -> Void)?
    var captureErrorHandler: ((String) -> Void)?

    func refreshWindows() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: true
        )
        let ownBundleID = Bundle.main.bundleIdentifier
        windows = content.windows.compactMap { window in
            guard window.isOnScreen,
                  window.windowLayer == 0,
                  let owner = window.owningApplication,
                  !owner.bundleIdentifier.isEmpty,
                  owner.bundleIdentifier != ownBundleID,
                  !owner.applicationName.isEmpty,
                  let application = NSRunningApplication(processIdentifier: owner.processID),
                  !application.isTerminated,
                  application.activationPolicy == .regular
            else {
                return nil
            }
            let windowTitle = window.title?.trimmingCharacters(in: .whitespacesAndNewlines)
            let title: String
            if let windowTitle, !windowTitle.isEmpty {
                title = windowTitle.localizedCaseInsensitiveCompare(owner.applicationName) == .orderedSame
                    ? owner.applicationName
                    : "\(owner.applicationName) — \(windowTitle)"
            } else {
                title = owner.applicationName
            }
            return ShareableWindow(id: window.windowID, title: title, nativeWindow: window)
        }.sorted { lhs, rhs in
            lhs.title.localizedStandardCompare(rhs.title) == .orderedAscending
        }
    }

    func start(windowID: UInt32) async throws {
        guard let target = windows.first(where: { $0.id == windowID }) else {
            throw CaptureError.windowUnavailable
        }
        faceTracks.removeAll()
        speakerTileState = nil
        stableVisibleFaceCount = nil
        pendingVisibleFaceCount = nil
        pendingVisibleFaceCountFrames = 0
        pendingSpeakerTileBounds.removeAll()
        collectingSpeakerTileLayout = true
        pendingSpeakerTileMisses = 0
        nextFaceTrackNumber = 1
        let filter = SCContentFilter(desktopIndependentWindow: target.nativeWindow)
        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.sampleRate = 16_000
        configuration.channelCount = 1
        let windowSize = target.nativeWindow.frame.size
        let captureScale = 640 / max(max(windowSize.width, windowSize.height), 1)
        configuration.width = max(2, Int((windowSize.width * captureScale).rounded()))
        configuration.height = max(2, Int((windowSize.height * captureScale).rounded()))
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 5)
        configuration.queueDepth = 3

        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        try stream.addStreamOutput(
            self,
            type: .audio,
            sampleHandlerQueue: DispatchQueue(label: "org.subtext.capture.audio")
        )
        try stream.addStreamOutput(
            self,
            type: .screen,
            sampleHandlerQueue: DispatchQueue(label: "org.subtext.capture.video")
        )
        try await stream.startCapture()
        self.stream = stream
        activeWindow = target.nativeWindow
        activeWindowProcessID = target.nativeWindow.owningApplication?.processID
        publishActiveWindowFrame()
        isCapturing = true
        errorMessage = nil
    }

    func stop() async throws {
        guard let stream else { return }
        try await stream.stopCapture()
        if let tail = audioBatcher.flush() {
            messageHandler?(Self.audioMessage(from: tail))
        }
        self.stream = nil
        activeWindow = nil
        activeWindowScreenFrame = nil
        activeWindowProcessID = nil
        windowFrameHandler?(nil, nil, nil)
        previewHandler?(nil)
        isCapturing = false
        faceTracks.removeAll()
        speakerTileState = nil
        stableVisibleFaceCount = nil
        pendingVisibleFaceCount = nil
        pendingVisibleFaceCountFrames = 0
        pendingSpeakerTileBounds.removeAll()
        collectingSpeakerTileLayout = true
        pendingSpeakerTileMisses = 0
    }

    nonisolated func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard sampleBuffer.isValid else { return }
        let capturedAt = Date().timeIntervalSince1970
        switch outputType {
        case .audio:
            guard let chunk = Self.audioPCMChunk(from: sampleBuffer, capturedAt: capturedAt) else { return }
            for readyChunk in audioBatcher.append(chunk) {
                Task { @MainActor [weak self] in
                    self?.messageHandler?(Self.audioMessage(from: readyChunk))
                }
            }
        case .screen:
            guard let output = Self.visualCapture(from: sampleBuffer, capturedAt: capturedAt) else { return }
            Task { @MainActor [weak self] in
                self?.publishActiveWindowFrame()
                self?.forwardVisualOutput(output)
            }
        case .microphone:
            return
        @unknown default:
            return
        }
    }

    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor [weak self] in
            self?.errorMessage = error.localizedDescription
            self?.isCapturing = false
            self?.stream = nil
            self?.activeWindow = nil
            self?.activeWindowScreenFrame = nil
            self?.activeWindowProcessID = nil
            self?.faceTracks.removeAll()
            self?.speakerTileState = nil
            self?.stableVisibleFaceCount = nil
            self?.pendingVisibleFaceCount = nil
            self?.pendingVisibleFaceCountFrames = 0
            self?.pendingSpeakerTileBounds.removeAll()
            self?.collectingSpeakerTileLayout = true
            self?.pendingSpeakerTileMisses = 0
            self?.windowFrameHandler?(nil, nil, nil)
            self?.previewHandler?(nil)
            self?.captureErrorHandler?(error.localizedDescription)
        }
    }

    private func forwardVisualOutput(
        _ output: (message: [String: Any], preview: LandmarkPreviewFrame?)
    ) {
        var message = output.message
        var preview = output.preview
        guard var subjects = message["subjects"] as? [[String: Any]] else {
            messageHandler?(message)
            previewHandler?(preview)
            return
        }
        let timestamp = message["timestamp"] as? TimeInterval ?? Date().timeIntervalSince1970
        var assigned = Set<String>()

        for index in subjects.indices {
            guard let values = subjects[index]["face_box"] as? [Double], values.count == 4 else {
                continue
            }
            let box = CGRect(
                x: CGFloat(values[0]),
                y: CGFloat(values[1]),
                width: CGFloat(values[2]),
                height: CGFloat(values[3])
            )
            var bestTrackID: String?
            var bestOverlap = 0.08
            for (trackID, previous) in faceTracks where !assigned.contains(trackID) {
                guard timestamp - previous.lastSeen <= 1.2 else { continue }
                let intersection = box.intersection(previous.boundingBox)
                guard !intersection.isNull else { continue }
                let intersectionArea = intersection.width * intersection.height
                let unionArea = box.width * box.height
                    + previous.boundingBox.width * previous.boundingBox.height
                    - intersectionArea
                guard unionArea > 0 else { continue }
                let overlap = intersectionArea / unionArea
                if overlap > bestOverlap {
                    bestOverlap = overlap
                    bestTrackID = trackID
                }
            }

            let trackID: String
            if let bestTrackID {
                trackID = bestTrackID
            } else {
                trackID = "window_face_\(nextFaceTrackNumber)"
                nextFaceTrackNumber += 1
            }
            assigned.insert(trackID)
            faceTracks[trackID] = FaceTrackState(boundingBox: box, lastSeen: timestamp)
            subjects[index]["track_id"] = trackID
            if let personName = subjects[index]["person_name"] as? String {
                personNameUpdateHandler?(trackID, personName)
            }
            subjects[index].removeValue(forKey: "person_name")
        }

        faceTracks = faceTracks.filter { timestamp - $0.value.lastSeen <= 1.2 }
        if var currentPreview = preview {
            if currentPreview.faces.isEmpty {
                pendingVisibleFaceCount = nil
                pendingVisibleFaceCountFrames = 0
                currentPreview.speakerTileBox = speakerTileState.map { normalizedBox(from: $0.bounds) }
                currentPreview.speakerTrackID = nil
            } else {
                let layoutChanged = updateStableVisibleFaceCount(currentPreview.faces.count)
                if let faceIndex = currentPreview.speakerFaceIndex,
                   subjects.indices.contains(faceIndex),
                   currentPreview.faces.indices.contains(faceIndex),
                   let trackID = subjects[faceIndex]["track_id"] as? String {
                    let faceBox = normalizedRect(from: currentPreview.faces[faceIndex].boundingBox)
                    let detectedBox = normalizedRect(from: currentPreview.speakerTileBox)
                    let lockedTile = resolveSpeakerTileBounds(
                        detected: detectedBox,
                        face: faceBox,
                        trackID: trackID,
                        imageAspectRatio: CGFloat(currentPreview.imageAspectRatio),
                        layoutChanged: layoutChanged
                    )
                    currentPreview.speakerTrackID = lockedTile == nil ? nil : trackID
                    currentPreview.speakerTileBox = lockedTile.map { normalizedBox(from: $0.bounds) }
                } else if let lockedTile = speakerTileState {
                    currentPreview.speakerTrackID = lockedTile.trackID
                    currentPreview.speakerTileBox = normalizedBox(from: lockedTile.bounds)
                } else {
                    currentPreview.speakerTileBox = nil
                    currentPreview.speakerTrackID = nil
                }
            }
            preview = currentPreview
        }
        message["subjects"] = subjects
        messageHandler?(message)
        previewHandler?(preview)
    }

    private func updateStableVisibleFaceCount(_ count: Int) -> Bool {
        guard count > 0 else { return false }
        if count == stableVisibleFaceCount {
            pendingVisibleFaceCount = nil
            pendingVisibleFaceCountFrames = 0
            return false
        }

        if pendingVisibleFaceCount == count {
            pendingVisibleFaceCountFrames += 1
        } else {
            pendingVisibleFaceCount = count
            pendingVisibleFaceCountFrames = 1
        }

        // At the capture rate of 5 fps this requires about a second of consistent
        // observations, so a single missed face does not unlock the panel bounds.
        guard pendingVisibleFaceCountFrames >= 6 else { return false }
        stableVisibleFaceCount = count
        pendingVisibleFaceCount = nil
        pendingVisibleFaceCountFrames = 0
        return true
    }

    private func resolveSpeakerTileBounds(
        detected: CGRect?,
        face: CGRect?,
        trackID: String,
        imageAspectRatio: CGFloat,
        layoutChanged: Bool
    ) -> SpeakerTileState? {
        let previous = speakerTileState
        if layoutChanged, previous != nil {
            collectingSpeakerTileLayout = true
            pendingSpeakerTileBounds.removeAll()
            pendingSpeakerTileMisses = 0
        }

        let candidate = detected ?? ((previous == nil || collectingSpeakerTileLayout || layoutChanged)
            ? face.map { estimateSpeakerTileBounds(around: $0, imageAspectRatio: imageAspectRatio) }
            : nil)

        guard let candidate else {
            if collectingSpeakerTileLayout {
                pendingSpeakerTileMisses += 1
                if pendingSpeakerTileMisses > 2 {
                    pendingSpeakerTileBounds.removeAll()
                    pendingSpeakerTileMisses = 0
                    collectingSpeakerTileLayout = previous == nil
                }
            }
            if let previous {
                speakerTileState = previous
                return previous
            }
            return nil
        }

        pendingSpeakerTileMisses = 0
        if let previous, !collectingSpeakerTileLayout, !layoutChanged {
            guard Self.isSignificantTileLayoutChange(candidate, from: previous.bounds) else {
                pendingSpeakerTileBounds.removeAll()
                speakerTileState = previous
                return previous
            }

            // Vision can return slightly different rectangles as the video changes.
            // Treat a new tile position as real only after a stable cluster of frames.
            collectingSpeakerTileLayout = true
            pendingSpeakerTileBounds = [candidate]
            return previous
        }

        if let pendingMedian = Self.medianTileBounds(pendingSpeakerTileBounds),
           !Self.tileBoundsMatch(candidate, pendingMedian) {
            pendingSpeakerTileBounds = [candidate]
        } else {
            pendingSpeakerTileBounds.append(candidate)
        }

        guard pendingSpeakerTileBounds.count >= 6,
              let stableBounds = Self.medianTileBounds(pendingSpeakerTileBounds) else {
            return previous
        }

        pendingSpeakerTileBounds.removeAll()
        collectingSpeakerTileLayout = false
        if let previous,
           !Self.isSignificantTileLayoutChange(stableBounds, from: previous.bounds) {
            let retained = SpeakerTileState(
                trackID: previous.trackID,
                bounds: previous.bounds
            )
            speakerTileState = retained
            return retained
        }

        let locked = SpeakerTileState(
            trackID: trackID,
            bounds: stableBounds
        )
        speakerTileState = locked
        return locked
    }

    nonisolated private static func tileBoundsMatch(_ lhs: CGRect, _ rhs: CGRect) -> Bool {
        abs(lhs.midX - rhs.midX) <= 0.02
            && abs(lhs.midY - rhs.midY) <= 0.02
            && abs(lhs.width - rhs.width) <= 0.03
            && abs(lhs.height - rhs.height) <= 0.03
    }

    nonisolated private static func isSignificantTileLayoutChange(_ candidate: CGRect, from previous: CGRect) -> Bool {
        let centerShift = hypot(candidate.midX - previous.midX, candidate.midY - previous.midY)
        let widthChange = abs(candidate.width - previous.width) / max(previous.width, 0.001)
        let heightChange = abs(candidate.height - previous.height) / max(previous.height, 0.001)
        return centerShift >= 0.03 || max(widthChange, heightChange) >= 0.08
    }

    nonisolated private static func medianTileBounds(_ bounds: [CGRect]) -> CGRect? {
        guard !bounds.isEmpty else { return nil }
        func median(_ values: [CGFloat]) -> CGFloat {
            let sorted = values.sorted()
            let midpoint = sorted.count / 2
            if sorted.count.isMultiple(of: 2) {
                return (sorted[midpoint - 1] + sorted[midpoint]) / 2
            }
            return sorted[midpoint]
        }
        return CGRect(
            x: median(bounds.map(\.minX)),
            y: median(bounds.map(\.minY)),
            width: median(bounds.map(\.width)),
            height: median(bounds.map(\.height))
        )
    }

    private func estimateSpeakerTileBounds(around face: CGRect, imageAspectRatio: CGFloat) -> CGRect {
        let safeImageAspect = imageAspectRatio.isFinite && imageAspectRatio > 0 ? imageAspectRatio : 1
        let tileAspect = CGFloat(16.0 / 9.0)
        var height = max(face.height * 2.25, face.width * safeImageAspect * 2.0 / tileAspect)
        var width = height * tileAspect / safeImageAspect
        let fitScale = min(1, min(0.96 / max(width, 0.001), 0.96 / max(height, 0.001)))
        width *= fitScale
        height *= fitScale
        let x = min(max(face.midX - width / 2, 0.02), 0.98 - width)
        let y = min(max(face.midY - height / 2, 0.02), 0.98 - height)
        return CGRect(x: x, y: y, width: width, height: height)
    }

    private func publishActiveWindowFrame() {
        guard let activeWindow,
              let screenFrame = Self.appKitScreenFrame(for: activeWindow.frame),
              screenFrame != activeWindowScreenFrame else {
            return
        }
        activeWindowScreenFrame = screenFrame
        windowFrameHandler?(screenFrame, activeWindowProcessID, activeWindow.windowID)
    }

    private static func appKitScreenFrame(for quartzFrame: CGRect) -> CGRect? {
        let center = CGPoint(x: quartzFrame.midX, y: quartzFrame.midY)
        for screen in NSScreen.screens {
            guard let displayID = screen.deviceDescription[
                NSDeviceDescriptionKey("NSScreenNumber")
            ] as? CGDirectDisplayID else {
                continue
            }
            let displayFrame = CGDisplayBounds(displayID)
            guard displayFrame.contains(center) else { continue }
            return CGRect(
                x: screen.frame.minX + quartzFrame.minX - displayFrame.minX,
                y: screen.frame.maxY - quartzFrame.maxY + displayFrame.minY,
                width: quartzFrame.width,
                height: quartzFrame.height
            )
        }
        return nil
    }

    nonisolated private static func audioPCMChunk(
        from sampleBuffer: CMSampleBuffer,
        capturedAt: TimeInterval
    ) -> AudioPCMChunk? {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)?.pointee,
              let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer)
        else {
            return nil
        }
        let pcmFormat: String
        if streamDescription.mBitsPerChannel == 32,
           streamDescription.mFormatFlags & kAudioFormatFlagIsFloat != 0 {
            pcmFormat = "float32"
        } else if streamDescription.mBitsPerChannel == 16 {
            pcmFormat = "int16"
        } else {
            return nil
        }

        let length = CMBlockBufferGetDataLength(blockBuffer)
        guard length > 0, length <= 120_000 else { return nil }
        var samples = Data(count: length)
        let status = samples.withUnsafeMutableBytes { destination -> OSStatus in
            guard let baseAddress = destination.baseAddress else { return -1 }
            return CMBlockBufferCopyDataBytes(
                blockBuffer,
                atOffset: 0,
                dataLength: length,
                destination: baseAddress.assumingMemoryBound(to: Int8.self)
            )
        }
        guard status == noErr else { return nil }

        return AudioPCMChunk(
            timestamp: capturedAt,
            sampleRate: Int(streamDescription.mSampleRate),
            pcmFormat: pcmFormat,
            samples: samples
        )
    }

    nonisolated private static func audioMessage(from chunk: AudioPCMChunk) -> [String: Any] {
        [
            "type": "audio_chunk",
            "timestamp": chunk.timestamp,
            "sample_rate": chunk.sampleRate,
            "pcm_format": chunk.pcmFormat,
            "samples": chunk.samples.base64EncodedString()
        ]
    }

    nonisolated private static func visualCapture(
        from sampleBuffer: CMSampleBuffer,
        capturedAt: TimeInterval
    ) -> (message: [String: Any], preview: LandmarkPreviewFrame?)? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let request = VNDetectFaceLandmarksRequest()
        let rectangleRequest = VNDetectRectanglesRequest()
        let textRequest = VNRecognizeTextRequest()
        textRequest.recognitionLevel = .accurate
        textRequest.usesLanguageCorrection = false
        textRequest.minimumTextHeight = 0.01
        rectangleRequest.maximumObservations = 48
        rectangleRequest.minimumConfidence = 0.15
        rectangleRequest.minimumSize = 0.08
        rectangleRequest.minimumAspectRatio = 0.35
        rectangleRequest.quadratureTolerance = 40
        let shouldRecognizeText = Int(capturedAt) != Int(capturedAt - 0.2)
        do {
            var requests: [VNRequest] = [request, rectangleRequest]
            if shouldRecognizeText {
                requests.append(textRequest)
            }
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform(requests)
        } catch {
            return nil
        }

        let faces = (request.results ?? []).sorted {
            $0.boundingBox.minX < $1.boundingBox.minX
        }
        let faceBoxes = faces.map(\.boundingBox)
        let primaryFaceIndex = faceBoxes.indices.max { lhs, rhs in
            faceBoxes[lhs].width * faceBoxes[lhs].height < faceBoxes[rhs].width * faceBoxes[rhs].height
        }
        let imageAspectRatio = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            / CGFloat(max(CVPixelBufferGetHeight(pixelBuffer), 1))
        let speakerTile = primaryFaceIndex.flatMap { faceIndex in
            speakerTileBounds(
                around: faceBoxes,
                targetFaceIndex: faceIndex,
                rectangles: rectangleRequest.results ?? [],
                imageAspectRatio: imageAspectRatio
            )
        }
        let speakerFaceIndex = speakerTile?.faceIndex ?? primaryFaceIndex
        let speakerTileBox = speakerTile.map { normalizedBox(from: $0.bounds) }
        let previewFaces: [FaceLandmarkPreview] = faces.map { face in
            let box = face.boundingBox
            return FaceLandmarkPreview(
                boundingBox: [Double(box.minX), Double(box.minY), Double(box.width), Double(box.height)],
                points: landmarkPoints(for: face)
            )
        }
        let rectangles = rectangleRequest.results ?? []
        let recognizedText = textRequest.results ?? []
        var cropPayloadSize = 0
        let subjects: [[String: Any]] = faces.enumerated().map { index, face in
            let box = face.boundingBox
            let landmarks = face.landmarks
            var subject: [String: Any] = [
                "track_id": "window_face_\(index + 1)",
                "face_present": true,
                // Keep the payload's numeric type aligned with forwardVisualOutput's
                // [Double] cast so OCR-only fields are removed before backend validation.
                "face_box": [Double(box.minX), Double(box.minY), Double(box.width), Double(box.height)],
                "confidence": face.confidence
            ]
            let participantTile = speakerTileBounds(
                around: faceBoxes,
                targetFaceIndex: index,
                rectangles: rectangles,
                imageAspectRatio: imageAspectRatio
            )
            if let personName = personName(in: participantTile?.bounds, textObservations: recognizedText) {
                subject["person_name"] = personName
            }
            if let yaw = face.yaw?.doubleValue {
                subject["head_yaw"] = yaw * 180.0 / .pi
            }
            if let pitch = face.pitch?.doubleValue {
                subject["head_pitch"] = pitch * 180.0 / .pi
            }
            if let roll = face.roll?.doubleValue {
                subject["head_roll"] = roll * 180.0 / .pi
            }
            if let aperture = aperture(landmarks?.innerLips) {
                subject["mouth_aperture"] = aperture
            }
            if let aperture = aperture(landmarks?.leftEye) {
                subject["left_eye_aperture"] = aperture
            }
            if let aperture = aperture(landmarks?.rightEye) {
                subject["right_eye_aperture"] = aperture
            }
            if index < 4,
               let crop = faceCropJPEGData(from: pixelBuffer, boundingBox: box) {
                let encodedCrop = crop.base64EncodedString()
                if encodedCrop.utf8.count <= 38_000,
                   cropPayloadSize + encodedCrop.utf8.count <= 140_000 {
                    subject["face_crop_jpeg"] = encodedCrop
                    cropPayloadSize += encodedCrop.utf8.count
                }
            }
            return subject
        }

        let message: [String: Any] = [
            "type": "visual_frame",
            "id": UUID().uuidString,
            "timestamp": capturedAt,
            "subjects": subjects,
            "quality": [
                "processor": "Apple Vision",
                "sample_rate_fps": 5,
                "raw_frame_sent": false,
                "face_crops_sent_to_local_service": true
            ]
        ]
        let preview = jpegData(from: pixelBuffer).map {
            LandmarkPreviewFrame(
                jpegData: $0,
                faces: previewFaces,
                speakerTileBox: speakerTileBox,
                speakerFaceIndex: speakerFaceIndex,
                speakerTrackID: nil,
                imageAspectRatio: Double(imageAspectRatio)
            )
        }
        return (message, preview)
    }

    nonisolated private static func speakerTileBounds(
        around faceBoxes: [CGRect],
        targetFaceIndex: Int,
        rectangles: [VNRectangleObservation],
        imageAspectRatio: CGFloat
    ) -> SpeakerTileCandidate? {
        guard faceBoxes.indices.contains(targetFaceIndex), imageAspectRatio > 0 else { return nil }
        let targetFace = faceBoxes[targetFaceIndex]
        let targetPhysicalAspect = 16.0 / 9.0

        let candidates = rectangles.compactMap { observation -> SpeakerTileCandidate? in
            let box = observation.boundingBox
            guard box.width > 0, box.height > 0,
                  box.width * box.height < 0.92 else {
                return nil
            }

            let facesInside = faceBoxes.indices.filter { index in
                box.insetBy(dx: box.width * 0.006, dy: box.height * 0.006)
                    .contains(faceBoxes[index])
            }
            guard facesInside.count == 1,
                  facesInside[0] == targetFaceIndex,
                  box.width >= targetFace.width * 1.35,
                  box.height >= targetFace.height * 1.35 else {
                return nil
            }

            let physicalAspect = box.width / box.height * imageAspectRatio
            guard (0.7...2.6).contains(physicalAspect) else { return nil }

            let aspectFit = exp(-abs(log(physicalAspect / targetPhysicalAspect)) * 1.6)
            let faceAreaRatio = (targetFace.width * targetFace.height) / (box.width * box.height)
            let faceScaleFit = exp(-abs(log(max(faceAreaRatio, 0.001) / 0.13)) * 0.45)
            let centerDistance = hypot(
                (targetFace.midX - box.midX) / box.width,
                (targetFace.midY - box.midY) / box.height
            )
            let centerFit = exp(-centerDistance * 2)
            let area = box.width * box.height
            let broadWindowPenalty = max(0, (area - 0.62) / 0.38) * 0.25
            let confidence = Double(observation.confidence)
            let score = aspectFit * 0.40
                + faceScaleFit * 0.24
                + centerFit * 0.16
                + confidence * 0.15
                + min(area / 0.38, 1) * 0.05
                - broadWindowPenalty
            return SpeakerTileCandidate(bounds: box, faceIndex: targetFaceIndex, score: score)
        }

        return candidates.max { $0.score < $1.score }
    }

    nonisolated private static func personName(
        in tile: CGRect?,
        textObservations: [VNRecognizedTextObservation]
    ) -> String? {
        guard let tile, tile.width > 0, tile.height > 0 else { return nil }
        let labelBand = CGRect(
            x: tile.minX,
            y: tile.minY,
            width: tile.width,
            height: tile.height * 0.30
        )
        let excludedLabels: Set<String> = [
            "you", "more", "mute", "unmute", "leave", "captions", "chat",
            "participants", "raise hand", "share", "stop video", "turn on camera"
        ]

        let candidates = textObservations.compactMap { observation -> (name: String, score: CGFloat)? in
            let textBox = observation.boundingBox
            guard labelBand.contains(CGPoint(x: textBox.midX, y: textBox.midY)),
                  textBox.height <= tile.height * 0.18,
                  let rawName = observation.topCandidates(1).first?.string else {
                return nil
            }
            let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedName = name.lowercased().trimmingCharacters(in: .punctuationCharacters)
            guard !name.isEmpty,
                  name.count <= 48,
                  !excludedLabels.contains(normalizedName) else {
                return nil
            }
            let distanceFromBottom = (textBox.midY - tile.minY) / tile.height
            let leftAlignment = (textBox.midX - tile.minX) / tile.width
            let score = 1 - distanceFromBottom - max(0, leftAlignment - 0.75) * 0.2
            return (name, score)
        }
        return candidates.max { $0.score < $1.score }?.name
    }

    nonisolated private static func landmarkPoints(for face: VNFaceObservation) -> [[Double]] {
        guard let landmarks = face.landmarks else { return [] }
        let box = face.boundingBox
        let regions: [VNFaceLandmarkRegion2D?] = [
            landmarks.faceContour,
            landmarks.leftEyebrow,
            landmarks.rightEyebrow,
            landmarks.leftEye,
            landmarks.rightEye,
            landmarks.nose,
            landmarks.outerLips,
            landmarks.innerLips
        ]
        return regions.compactMap { $0 }.flatMap { region in
            region.normalizedPoints.map { point in
                [
                    Double(box.minX + point.x * box.width),
                    Double(box.minY + point.y * box.height)
                ]
            }
        }
    }

    nonisolated private static func jpegData(from pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }

        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            "public.jpeg" as CFString,
            1,
            nil
        ) else {
            return nil
        }
        CGImageDestinationAddImage(
            destination,
            cgImage,
            [kCGImageDestinationLossyCompressionQuality: 0.72] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }

    nonisolated private static func faceCropJPEGData(
        from pixelBuffer: CVPixelBuffer,
        boundingBox: CGRect
    ) -> Data? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = image.extent
        let faceRect = CGRect(
            x: extent.minX + boundingBox.minX * extent.width,
            y: extent.minY + boundingBox.minY * extent.height,
            width: boundingBox.width * extent.width,
            height: boundingBox.height * extent.height
        )
        let side = max(faceRect.width, faceRect.height) * 1.35
        guard side > 1 else { return nil }
        let requestedRect = CGRect(
            x: faceRect.midX - side / 2,
            y: faceRect.midY - side / 2,
            width: side,
            height: side
        )
        let cropRect = requestedRect.intersection(extent)
        guard cropRect.width > 1, cropRect.height > 1 else { return nil }

        let imageContext = CIContext()
        guard let croppedImage = imageContext.createCGImage(image, from: cropRect),
              let bitmap = CGContext(
                data: nil,
                width: 224,
                height: 224,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              )
        else {
            return nil
        }
        bitmap.interpolationQuality = .high
        bitmap.draw(croppedImage, in: CGRect(x: 0, y: 0, width: 224, height: 224))
        guard let resizedImage = bitmap.makeImage() else { return nil }

        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            "public.jpeg" as CFString,
            1,
            nil
        ) else {
            return nil
        }
        CGImageDestinationAddImage(
            destination,
            resizedImage,
            [kCGImageDestinationLossyCompressionQuality: 0.48] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }

    nonisolated private static func aperture(_ region: VNFaceLandmarkRegion2D?) -> Double? {
        guard let points = region?.normalizedPoints, points.count >= 3 else { return nil }
        let xs = points.map(\.x)
        let ys = points.map(\.y)
        guard let minX = xs.min(), let maxX = xs.max(),
              let minY = ys.min(), let maxY = ys.max()
        else {
            return nil
        }
        let width = maxX - minX
        guard width > 0.0001 else { return nil }
        return Double((maxY - minY) / width)
    }
}

private enum CaptureError: LocalizedError {
    case windowUnavailable

    var errorDescription: String? {
        "That call window is no longer available. Refresh the window list and try again."
    }
}
