import XCTest
@testable import CaptureSupport

final class AudioChunkBatcherTests: XCTestCase {
    func testCombinesSmallFloat32CallbacksIntoHundredMillisecondChunks() {
        let batcher = AudioChunkBatcher(targetDuration: 0.1)
        let samplesPerCallback = 320
        let callback = Data(repeating: 0x2a, count: samplesPerCallback * MemoryLayout<Float>.size)
        var output: [AudioPCMChunk] = []

        for index in 0..<5 {
            output += batcher.append(AudioPCMChunk(
                timestamp: 10 + Double(index) * 0.02,
                sampleRate: 16_000,
                pcmFormat: "float32",
                samples: callback
            ))
        }

        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].timestamp, 10, accuracy: 0.0001)
        XCTAssertEqual(output[0].samples, Data(repeating: 0x2a, count: 1_600 * 4))
        XCTAssertNil(batcher.flush())
    }

    func testFlushReturnsPartialInt16TailWithOriginalTimestamp() {
        let batcher = AudioChunkBatcher(targetDuration: 0.1)
        let samples = Data(repeating: 0x17, count: 400 * MemoryLayout<Int16>.size)

        XCTAssertTrue(batcher.append(AudioPCMChunk(
            timestamp: 24.5,
            sampleRate: 16_000,
            pcmFormat: "int16",
            samples: samples
        )).isEmpty)

        let tail = batcher.flush()
        XCTAssertEqual(tail?.timestamp, 24.5)
        XCTAssertEqual(tail?.samples, samples)
        XCTAssertNil(batcher.flush())
    }

    func testFormatChangeFlushesPreviousAudioBeforeNewFormat() {
        let batcher = AudioChunkBatcher(targetDuration: 0.1)
        let floatSamples = Data(repeating: 0x11, count: 400 * 4)
        let intSamples = Data(repeating: 0x22, count: 400 * 2)

        _ = batcher.append(AudioPCMChunk(
            timestamp: 30,
            sampleRate: 16_000,
            pcmFormat: "float32",
            samples: floatSamples
        ))
        let output = batcher.append(AudioPCMChunk(
            timestamp: 30.03,
            sampleRate: 16_000,
            pcmFormat: "int16",
            samples: intSamples
        ))

        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].pcmFormat, "float32")
        XCTAssertEqual(output[0].samples, floatSamples)
        XCTAssertEqual(batcher.flush()?.pcmFormat, "int16")
    }

    func testDiscontinuousAudioDoesNotHoldOldTailUntilLaterSpeech() {
        let batcher = AudioChunkBatcher(targetDuration: 0.1)
        let samples = Data(repeating: 0x33, count: 400 * 4)

        _ = batcher.append(AudioPCMChunk(
            timestamp: 40,
            sampleRate: 16_000,
            pcmFormat: "float32",
            samples: samples
        ))
        let output = batcher.append(AudioPCMChunk(
            timestamp: 40.3,
            sampleRate: 16_000,
            pcmFormat: "float32",
            samples: samples
        ))

        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].timestamp, 40, accuracy: 0.0001)
        XCTAssertEqual(output[0].samples, samples)
        let tail = batcher.flush()
        XCTAssertEqual(tail?.timestamp ?? 0, 40.3, accuracy: 0.0001)
    }
}
