import Foundation

public struct AudioPCMChunk: Sendable, Equatable {
    public let timestamp: TimeInterval
    public let sampleRate: Int
    public let pcmFormat: String
    public let samples: Data

    public init(timestamp: TimeInterval, sampleRate: Int, pcmFormat: String, samples: Data) {
        self.timestamp = timestamp
        self.sampleRate = sampleRate
        self.pcmFormat = pcmFormat
        self.samples = samples
    }
}

/// Combines small capture callbacks into fewer websocket packets while keeping
/// the original PCM bytes and sample timeline intact.
public final class AudioChunkBatcher: @unchecked Sendable {
    private let lock = NSLock()
    private let targetDuration: TimeInterval
    private var pending = Data()
    private var pendingTimestamp: TimeInterval?
    private var pendingSampleRate: Int?
    private var pendingPCMFormat: String?

    public init(targetDuration: TimeInterval = 0.1) {
        self.targetDuration = max(0.02, targetDuration)
    }

    /// Returns complete chunks once enough samples have accumulated.
    public func append(_ chunk: AudioPCMChunk) -> [AudioPCMChunk] {
        lock.lock()
        defer { lock.unlock() }

        guard chunk.sampleRate > 0,
              let bytesPerSample = Self.bytesPerSample(for: chunk.pcmFormat),
              !chunk.samples.isEmpty,
              chunk.samples.count.isMultiple(of: bytesPerSample)
        else {
            return []
        }

        var ready: [AudioPCMChunk] = []
        if let pendingTimestamp,
           let pendingSampleRate,
           let pendingPCMFormat,
           (pendingSampleRate != chunk.sampleRate
            || pendingPCMFormat != chunk.pcmFormat
            || chunk.timestamp - pendingTimestamp - Double(pending.count / bytesPerSample) / Double(pendingSampleRate) > targetDuration * 2) {
            if let previous = takePending() {
                ready.append(previous)
            }
        }

        if pendingTimestamp == nil {
            pendingTimestamp = chunk.timestamp
            pendingSampleRate = chunk.sampleRate
            pendingPCMFormat = chunk.pcmFormat
        }

        let targetBytes = max(
            bytesPerSample,
            Int((Double(chunk.sampleRate) * targetDuration).rounded()) * bytesPerSample
        )
        var consumedBytes = 0
        while consumedBytes < chunk.samples.count {
            let capacity = max(0, targetBytes - pending.count)
            let bytesToCopy = min(capacity, chunk.samples.count - consumedBytes)
            if bytesToCopy > 0 {
                pending.append(chunk.samples.subdata(in: consumedBytes..<(consumedBytes + bytesToCopy)))
                consumedBytes += bytesToCopy
            }

            if pending.count >= targetBytes, let complete = takePending() {
                ready.append(complete)
                pendingTimestamp = chunk.timestamp + Double(consumedBytes / bytesPerSample) / Double(chunk.sampleRate)
                pendingSampleRate = chunk.sampleRate
                pendingPCMFormat = chunk.pcmFormat
            }
        }

        // Avoid retaining a timestamp for a zero-length tail after an exact split.
        if pending.isEmpty {
            pendingTimestamp = nil
            pendingSampleRate = nil
            pendingPCMFormat = nil
        }
        return ready
    }

    /// Emits the final partial chunk when capture stops.
    public func flush() -> AudioPCMChunk? {
        lock.lock()
        defer { lock.unlock() }
        return takePending()
    }

    private func takePending() -> AudioPCMChunk? {
        guard !pending.isEmpty,
              let timestamp = pendingTimestamp,
              let sampleRate = pendingSampleRate,
              let pcmFormat = pendingPCMFormat else {
            pending.removeAll(keepingCapacity: true)
            pendingTimestamp = nil
            pendingSampleRate = nil
            pendingPCMFormat = nil
            return nil
        }
        let chunk = AudioPCMChunk(
            timestamp: timestamp,
            sampleRate: sampleRate,
            pcmFormat: pcmFormat,
            samples: pending
        )
        pending.removeAll(keepingCapacity: true)
        pendingTimestamp = nil
        pendingSampleRate = nil
        pendingPCMFormat = nil
        return chunk
    }

    private static func bytesPerSample(for pcmFormat: String) -> Int? {
        switch pcmFormat {
        case "float32": 4
        case "int16": 2
        default: nil
        }
    }
}
