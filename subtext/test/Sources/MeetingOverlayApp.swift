import AVFoundation
import AppKit
import AudioToolbox
import Combine
import CoreMedia
import Foundation
import ScreenCaptureKit
import SwiftUI

@main
struct MeetingOverlayApp: App {
    @NSApplicationDelegateAdaptor(OverlayAppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings { EmptyView() }
    }
}

@MainActor
final class OverlayAppDelegate: NSObject, NSApplicationDelegate {
    private let client = TranscriptionClient()
    private var panel: NSPanel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 380, height: 570),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.title = "Meeting Overlay Test"
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
                NSPoint(x: frame.maxX - panel.frame.width - 24, y: frame.maxY - panel.frame.height - 36)
            )
        }

        self.panel = panel
        panel.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)
        client.connect()
    }
}

struct TranscriptLine: Identifiable {
    let id: String
    let text: String
    let time: Date
    let speakerID: String
}

private struct ServiceEvent: Decodable {
    let type: String
    let state: String?
    let detail: String?
    let id: String?
    let text: String?
    let timestamp: Double?
    let speakerId: String?
}

private struct ServiceError: Decodable {
    let detail: String
}

@MainActor
final class TranscriptionClient: ObservableObject {
    @Published private(set) var transcript: [TranscriptLine] = []
    @Published private(set) var windows: [MeetingWindow] = []
    @Published var selectedWindowID: UInt32?
    @Published private(set) var stateLabel = "Connecting to local transcription"
    @Published private(set) var isConnected = false
    @Published private(set) var isListening = false
    @Published private(set) var isStarting = false
    @Published private(set) var microphoneLevel = 0.0
    @Published private(set) var errorMessage: String?

    private let serviceURL = URL(string: "http://127.0.0.1:8765")!
    private var receiveTask: Task<Void, Never>?
    private var socket: URLSessionWebSocketTask?
    private var ingestSocket: URLSessionWebSocketTask?
    private var captureSendTask: Task<Void, Never>?
    private var captureQueue: [String] = []
    private var isCapturingMeetingAudio = false
    private let microphone = MicrophoneCapture()
    private let windowCapture = MeetingWindowCapture()

    init() {
        microphone.messageHandler = { [weak self] message in
            Task { @MainActor in self?.sendCaptureMessage(message) }
        }
        microphone.levelHandler = { [weak self] level in
            Task { @MainActor in self?.microphoneLevel = level }
        }
        windowCapture.messageHandler = { [weak self] message in
            self?.sendCaptureMessage(message)
        }
        windowCapture.errorHandler = { [weak self] detail in
            self?.handleCaptureFailure(detail)
        }
    }

    func connect() {
        guard receiveTask == nil else { return }
        receiveTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                let candidate = URLSession.shared.webSocketTask(
                    with: URL(string: "ws://127.0.0.1:8765/ws")!
                )
                socket = candidate
                candidate.resume()
                stateLabel = "Connecting to local transcription"
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
                    candidate.cancel(with: .goingAway, reason: nil)
                }
                guard !Task.isCancelled else { break }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    func refreshWindows() async {
        do {
            try await windowCapture.refreshWindows()
            windows = windowCapture.windows
            if let selectedWindowID, !windows.contains(where: { $0.id == selectedWindowID }) {
                self.selectedWindowID = nil
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func start() async {
        errorMessage = nil
        isStarting = true
        stateLabel = "Starting transcription"
        microphoneLevel = 0
        let captureSocket = URLSession.shared.webSocketTask(
            with: URL(string: "ws://127.0.0.1:8765/ingest")!
        )
        ingestSocket = captureSocket
        captureSocket.resume()

        do {
            try await post(path: "/start")
            try await microphone.start()
            var capturesMeetingAudio = false
            if let selectedWindowID {
                do {
                    try await windowCapture.start(windowID: selectedWindowID)
                    capturesMeetingAudio = true
                } catch {
                    errorMessage = "Your microphone is active, but the meeting window could not be captured: \(error.localizedDescription)"
                }
            }
            isCapturingMeetingAudio = capturesMeetingAudio
            isStarting = false
            isListening = true
            stateLabel = capturesMeetingAudio ? "Listening to both sides" : "Listening to your microphone"
        } catch {
            isStarting = false
            isListening = false
            microphone.stop()
            isCapturingMeetingAudio = false
            try? await windowCapture.stop()
            captureSocket.cancel(with: .goingAway, reason: nil)
            ingestSocket = nil
            try? await post(path: "/stop")
            errorMessage = error.localizedDescription
            stateLabel = "Could not start"
        }
    }

    func stop() async {
        isStarting = false
        errorMessage = nil
        microphone.stop()
        microphoneLevel = 0
        isCapturingMeetingAudio = false
        do {
            try await windowCapture.stop()
            try await post(path: "/stop")
            ingestSocket?.cancel(with: .goingAway, reason: nil)
            ingestSocket = nil
            captureQueue.removeAll()
            isListening = false
            stateLabel = "Ready"
        } catch {
            errorMessage = error.localizedDescription
            stateLabel = "Could not stop"
        }
    }

    func clearTranscript() {
        transcript.removeAll()
        Task {
            var request = URLRequest(url: serviceURL.appending(path: "/transcript"))
            request.httpMethod = "DELETE"
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    private func post(path: String) async throws {
        var request = URLRequest(url: serviceURL.appending(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = 240
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ClientError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            let detail = (try? JSONDecoder().decode(ServiceError.self, from: data).detail)
                ?? "The local transcription service returned an error."
            throw ClientError.service(detail)
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        let data: Data
        switch message {
        case .data(let received): data = received
        case .string(let received): data = Data(received.utf8)
        @unknown default: return
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let event = try? decoder.decode(ServiceEvent.self, from: data) else { return }

        if event.type == "cleared" {
            transcript.removeAll()
        } else if event.type == "transcript", let text = event.text, !text.isEmpty {
            transcript.append(
                TranscriptLine(
                    id: event.id ?? UUID().uuidString,
                    text: text,
                    time: Date(timeIntervalSince1970: event.timestamp ?? Date().timeIntervalSince1970),
                    speakerID: event.speakerId ?? "self"
                )
            )
            if transcript.count > 80 { transcript.removeFirst(transcript.count - 80) }
        } else if event.type == "status", let state = event.state {
            switch state {
            case "idle":
                isListening = false
                isStarting = false
                stateLabel = "Ready"
            case "loading_model":
                isStarting = true
                stateLabel = "Loading transcription model"
            case "listening":
                isListening = true
                isStarting = false
                stateLabel = isCapturingMeetingAudio ? "Listening to both sides" : "Listening to your microphone"
            case "transcribing":
                isListening = true
                stateLabel = isCapturingMeetingAudio ? "Transcribing both sides" : "Transcribing your microphone"
            case "error":
                isListening = false
                isStarting = false
                stateLabel = "Needs attention"
                errorMessage = event.detail ?? "The transcription service hit an error."
            default:
                stateLabel = state.replacingOccurrences(of: "_", with: " ").capitalized
            }
        }
    }

    private func handleCaptureFailure(_ detail: String) {
        isCapturingMeetingAudio = false
        errorMessage = "Meeting-window audio stopped. Your microphone is still listening. \(detail)"
        stateLabel = "Listening to your microphone"
    }

    private func sendCaptureMessage(_ message: [String: Any]) {
        guard ingestSocket != nil,
              let data = try? JSONSerialization.data(withJSONObject: message),
              let text = String(data: data, encoding: .utf8)
        else { return }
        captureQueue.append(text)
        if captureQueue.count > 80 { captureQueue.removeFirst(captureQueue.count - 80) }
        guard captureSendTask == nil else { return }
        captureSendTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled, !captureQueue.isEmpty {
                guard let socket = ingestSocket else { break }
                let item = captureQueue.removeFirst()
                do {
                    try await socket.send(.string(item))
                } catch {
                    captureQueue.removeAll()
                    break
                }
            }
            captureSendTask = nil
        }
    }
}

private enum ClientError: LocalizedError {
    case invalidResponse
    case service(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse: "The local transcription service returned an invalid response."
        case .service(let message): message
        }
    }
}

struct OverlayView: View {
    @ObservedObject var client: TranscriptionClient

    private var selectedWindow: MeetingWindow? {
        client.windows.first(where: { $0.id == client.selectedWindowID })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().overlay(Color.white.opacity(0.10)).padding(.vertical, 15)
            sourcePicker
            microphoneMeter
            transcriptSection
            if let error = client.errorMessage {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundStyle(Color.orange.opacity(0.95))
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 9)
            }
            Spacer(minLength: 13)
            listenButton
        }
        .foregroundStyle(.white)
        .padding(19)
        .frame(width: 380, height: 570)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
        .overlay {
            RoundedRectangle(cornerRadius: 24)
                .stroke(Color.white.opacity(0.13), lineWidth: 1)
        }
        .onAppear { Task { await client.refreshWindows() } }
    }

    private var header: some View {
        HStack(spacing: 11) {
            Image(systemName: "waveform")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 38, height: 38)
                .background(Color(red: 0.31, green: 0.47, blue: 0.94), in: RoundedRectangle(cornerRadius: 13))

            VStack(alignment: .leading, spacing: 3) {
                Text("Meeting transcript")
                    .font(.system(size: 16, weight: .semibold))
                HStack(spacing: 6) {
                    Circle()
                        .fill(client.isListening ? Color.green : Color.white.opacity(0.42))
                        .frame(width: 6, height: 6)
                    Text(client.stateLabel)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.white.opacity(0.62))
                }
            }
            Spacer()
            Button(action: client.clearTranscript) {
                Image(systemName: "trash")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.white.opacity(0.68))
                    .frame(width: 29, height: 29)
                    .background(Color.white.opacity(0.08), in: Circle())
            }
            .buttonStyle(.plain)
            .help("Clear transcript")
            Button(action: { NSApp.terminate(nil) }) {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.white.opacity(0.68))
                    .frame(width: 29, height: 29)
                    .background(Color.white.opacity(0.08), in: Circle())
            }
            .buttonStyle(.plain)
            .help("Quit overlay")
        }
    }

    private var sourcePicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("MEETING WINDOW OR TAB")
                    .font(.system(size: 9, weight: .semibold))
                    .tracking(1.0)
                    .foregroundStyle(.white.opacity(0.50))
                Spacer()
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
                .help("Refresh available windows")
                .disabled(client.isListening || client.isStarting)
            }

            Picker("Meeting window", selection: $client.selectedWindowID) {
                Text("Microphone only").tag(nil as UInt32?)
                ForEach(client.windows) { window in
                    Text("\(window.applicationName) — \(window.title)")
                        .tag(Optional(window.id))
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .tint(.white)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 11)
            .frame(height: 40)
            .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
            .disabled(client.isListening || client.isStarting || client.windows.isEmpty)

            if let selectedWindow {
                Text("\(selectedWindow.applicationName) · \(selectedWindow.title)")
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.47))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("Microphone and meeting audio are transcribed locally.")
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.42))
            } else {
                Text("Your microphone works on its own. Select a window to add the other person.")
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.47))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.bottom, 14)
    }

    private var transcriptSection: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text("TRANSCRIPT")
                    .font(.system(size: 9, weight: .semibold))
                    .tracking(1.2)
                    .foregroundStyle(.white.opacity(0.50))
                Spacer()
                Text("YOU  +  OTHER PERSON")
                    .font(.system(size: 8, weight: .medium))
                    .tracking(0.7)
                    .foregroundStyle(.white.opacity(0.38))
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 9) {
                        if client.transcript.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Image(systemName: client.isListening ? "waveform" : "text.bubble")
                                    .font(.system(size: 17, weight: .medium))
                                    .foregroundStyle(.white.opacity(0.72))
                                    .padding(.bottom, 2)
                                Text(client.isListening ? "Listening for conversation…" : "Your conversation will appear here.")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundStyle(.white.opacity(0.82))
                                Text("Start to transcribe your microphone, or select a window to include the other person.")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.white.opacity(0.48))
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(13)
                            .padding(.top, 8)
                        } else {
                            ForEach(client.transcript) { line in
                                TranscriptBubble(line: line)
                                    .id(line.id)
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
            .frame(maxWidth: .infinity)
            .frame(height: 275)
            .background(Color.black.opacity(0.14), in: RoundedRectangle(cornerRadius: 16))
        }
    }

    private var microphoneMeter: some View {
        let decibels = 20 * log10(max(client.microphoneLevel, 0.00001))
        let fill = min(1, max(0, (decibels + 60) / 60))
        return HStack(spacing: 8) {
            Image(systemName: "mic.fill")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white.opacity(0.58))
            Text("MICROPHONE")
                .font(.system(size: 8, weight: .semibold))
                .tracking(0.8)
                .foregroundStyle(.white.opacity(0.48))
            Spacer()
            Text(client.isListening && client.microphoneLevel > 0.002 ? "Input detected" : "Waiting for sound")
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
        }
        .frame(height: 14)
        .padding(.bottom, 13)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Microphone level")
    }

    private var listenButton: some View {
        Button {
            Task {
                if client.isListening { await client.stop() }
                else { await client.start() }
            }
        } label: {
            HStack(spacing: 9) {
                Image(systemName: client.isListening ? "stop.fill" : "mic.fill")
                    .font(.system(size: 12, weight: .semibold))
                Text(client.isStarting ? "Starting…" : (client.isListening ? "Stop transcription" : "Start transcription"))
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
            .frame(height: 44)
            .background(
                client.isListening ? Color.white.opacity(0.16) : Color(red: 0.31, green: 0.47, blue: 0.94),
                in: RoundedRectangle(cornerRadius: 13)
            )
        }
        .buttonStyle(.plain)
        .disabled(!client.isConnected || client.isStarting)
        .opacity(client.isConnected && !client.isStarting ? 1 : 0.56)
    }
}

private struct TranscriptBubble: View {
    let line: TranscriptLine

    private var isSelf: Bool { line.speakerID == "self" }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Circle()
                    .fill(isSelf ? Color(red: 0.58, green: 0.69, blue: 1.0) : Color(red: 0.45, green: 0.82, blue: 0.71))
                    .frame(width: 5, height: 5)
                Text(isSelf ? "You" : "Other person")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.72))
                Text(line.time.formatted(date: .omitted, time: .shortened))
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.white.opacity(0.39))
            }
            Text(line.text)
                .font(.system(size: 12.5))
                .foregroundStyle(.white.opacity(0.94))
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 11)
        .padding(.vertical, 9)
        .background(
            isSelf ? Color(red: 0.25, green: 0.34, blue: 0.58).opacity(0.42) : Color.white.opacity(0.07),
            in: RoundedRectangle(cornerRadius: 12)
        )
        .padding(.horizontal, 8)
    }
}

struct MeetingWindow: Identifiable {
    let id: UInt32
    let applicationName: String
    let title: String
    fileprivate let nativeWindow: SCWindow
}

@MainActor
final class MeetingWindowCapture: NSObject, ObservableObject, SCStreamOutput, SCStreamDelegate {
    @Published private(set) var windows: [MeetingWindow] = []
    private var stream: SCStream?
    var messageHandler: (([String: Any]) -> Void)?
    var errorHandler: ((String) -> Void)?

    func refreshWindows() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        let appBundleID = Bundle.main.bundleIdentifier
        windows = content.windows.compactMap { window in
            guard window.isOnScreen, window.owningApplication?.bundleIdentifier != appBundleID else { return nil }
            let appName = window.owningApplication?.applicationName ?? "App"
            let title = window.title?.isEmpty == false ? window.title! : appName
            return MeetingWindow(id: window.windowID, applicationName: appName, title: title, nativeWindow: window)
        }
        .sorted {
            if $0.applicationName == $1.applicationName { return $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
            return $0.applicationName.localizedCaseInsensitiveCompare($1.applicationName) == .orderedAscending
        }
    }

    func start(windowID: UInt32) async throws {
        guard let target = windows.first(where: { $0.id == windowID }) else {
            throw CaptureError.windowUnavailable
        }
        let filter = SCContentFilter(desktopIndependentWindow: target.nativeWindow)
        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.sampleRate = 16_000
        configuration.channelCount = 1
        configuration.width = 2
        configuration.height = 2
        configuration.queueDepth = 3
        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        try stream.addStreamOutput(
            self,
            type: .audio,
            sampleHandlerQueue: DispatchQueue(label: "org.subtext.meeting-overlay.audio")
        )
        try await stream.startCapture()
        self.stream = stream
    }

    func stop() async throws {
        guard let stream else { return }
        try await stream.stopCapture()
        self.stream = nil
    }

    nonisolated func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio, sampleBuffer.isValid,
              let message = Self.audioMessage(from: sampleBuffer, capturedAt: Date().timeIntervalSince1970)
        else { return }
        Task { @MainActor [weak self] in self?.messageHandler?(message) }
    }

    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor [weak self] in
            self?.stream = nil
            self?.errorHandler?(error.localizedDescription)
        }
    }

    nonisolated private static func audioMessage(
        from sampleBuffer: CMSampleBuffer,
        capturedAt: TimeInterval
    ) -> [String: Any]? {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)?.pointee,
              let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer)
        else { return nil }

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
        return [
            "type": "audio_chunk",
            "timestamp": capturedAt,
            "sample_rate": Int(streamDescription.mSampleRate),
            "pcm_format": pcmFormat,
            "samples": samples.base64EncodedString()
        ]
    }
}

private enum CaptureError: LocalizedError {
    case windowUnavailable

    var errorDescription: String? {
        "That meeting window is no longer available. Refresh the list and try again."
    }
}

final class MicrophoneCapture {
    private let sampleRate = 16_000.0
    private let engine = AVAudioEngine()
    private var inputNode: AVAudioInputNode?
    private var converter: AVAudioConverter?
    private var outputFormat: AVAudioFormat?
    private(set) var isCapturing = false
    var messageHandler: (([String: Any]) -> Void)?
    var levelHandler: ((Double) -> Void)?

    func start() async throws {
        guard !isCapturing else { return }
        guard await microphonePermissionGranted() else { throw MicrophoneError.permissionDenied }
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else { throw MicrophoneError.inputUnavailable }
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ), let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw MicrophoneError.converterUnavailable
        }
        converter.downmix = true

        self.inputNode = inputNode
        self.converter = converter
        self.outputFormat = outputFormat
        inputNode.installTap(onBus: 0, bufferSize: 4_096, format: inputFormat) { [weak self] buffer, _ in
            self?.convertAndSend(buffer)
        }
        engine.prepare()
        do {
            try engine.start()
            isCapturing = true
        } catch {
            inputNode.removeTap(onBus: 0)
            self.inputNode = nil
            self.converter = nil
            self.outputFormat = nil
            throw error
        }
    }

    func stop() {
        guard isCapturing else { return }
        inputNode?.removeTap(onBus: 0)
        engine.stop()
        inputNode = nil
        converter = nil
        outputFormat = nil
        isCapturing = false
    }

    private func microphonePermissionGranted() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        case .denied, .restricted: return false
        @unknown default: return false
        }
    }

    private func convertAndSend(_ inputBuffer: AVAudioPCMBuffer) {
        guard inputBuffer.frameLength > 0, let converter, let outputFormat else { return }
        let ratio = outputFormat.sampleRate / inputBuffer.format.sampleRate
        let capacity = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio) + 32)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }
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
        guard status != .error, outputBuffer.frameLength > 0, let channels = outputBuffer.floatChannelData else { return }
        let byteCount = Int(outputBuffer.frameLength) * MemoryLayout<Float>.size
        let frameCount = Int(outputBuffer.frameLength)
        var sumSquares = 0.0
        for index in 0..<frameCount {
            let sample = Double(channels[0][index])
            sumSquares += sample * sample
        }
        levelHandler?(sqrt(sumSquares / Double(max(frameCount, 1))))
        let pcm = Data(bytes: channels[0], count: byteCount)
        messageHandler?([
            "type": "microphone_chunk",
            "timestamp": Date().timeIntervalSince1970,
            "sample_rate": Int(sampleRate),
            "pcm_format": "float32",
            "samples": pcm.base64EncodedString()
        ])
    }
}

private enum MicrophoneError: LocalizedError {
    case permissionDenied
    case inputUnavailable
    case converterUnavailable

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            "Allow microphone access in System Settings → Privacy & Security → Microphone, then restart the overlay."
        case .inputUnavailable:
            "No microphone input is available. Choose one in System Settings → Sound → Input."
        case .converterUnavailable:
            "The microphone audio format could not be prepared."
        }
    }
}
