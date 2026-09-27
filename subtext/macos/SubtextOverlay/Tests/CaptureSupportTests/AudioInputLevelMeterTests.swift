import AVFoundation
import AudioToolbox
import XCTest
@testable import CaptureSupport

final class AudioInputLevelMeterTests: XCTestCase {
    func testOppositePhaseMicrophoneChannelsDoNotCancel() throws {
        let buffer = try makeFloatBuffer(channels: 2, interleaved: false, frames: 4)
        let channels = try XCTUnwrap(buffer.floatChannelData)
        let values: [Float] = [0.25, -0.25, 0.25, -0.25]
        for frame in values.indices {
            channels[0][frame] = values[frame]
            channels[1][frame] = -values[frame]
        }

        let levels = try XCTUnwrap(AudioInputLevelMeter.channelRMSLevels(in: buffer))
        XCTAssertEqual(levels.count, 2)
        XCTAssertEqual(levels[0], 0.25, accuracy: 0.0001)
        XCTAssertEqual(levels[1], 0.25, accuracy: 0.0001)
    }

    func testInterleavedChannelsAreMeasuredIndependently() throws {
        let buffer = try makeFloatBuffer(channels: 2, interleaved: true, frames: 2)
        let interleaved = try XCTUnwrap(buffer.floatChannelData)[0]
        interleaved[0] = 0.5
        interleaved[1] = 0.0
        interleaved[2] = -0.5
        interleaved[3] = 0.0

        let levels = try XCTUnwrap(AudioInputLevelMeter.channelRMSLevels(in: buffer))
        XCTAssertEqual(levels[0], 0.5, accuracy: 0.0001)
        XCTAssertEqual(levels[1], 0.0, accuracy: 0.0001)
    }

    func testResamplingMapsTheStrongestInputChannelWithoutDownmixing() throws {
        let input = try makeFloatBuffer(
            channels: 2,
            interleaved: false,
            frames: 4_800,
            sampleRate: 48_000
        )
        let inputChannels = try XCTUnwrap(input.floatChannelData)
        for frame in 0..<Int(input.frameLength) {
            let sample = Float(0.25 * sin(2 * Double.pi * 440 * Double(frame) / 48_000))
            inputChannels[0][frame] = sample
            inputChannels[1][frame] = -sample
        }

        let outputFormat = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16_000,
                channels: 1,
                interleaved: false
            )
        )
        let converter = try XCTUnwrap(AVAudioConverter(from: input.format, to: outputFormat))
        converter.downmix = false
        converter.channelMap = [NSNumber(value: 1)]
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) / 3) + 32)
        let output = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity)
        )
        var suppliedInput = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
            guard !suppliedInput else {
                inputStatus.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            inputStatus.pointee = .haveData
            return input
        }

        XCTAssertNotEqual(status, .error, conversionError?.localizedDescription ?? "conversion failed")
        let levels = try XCTUnwrap(AudioInputLevelMeter.channelRMSLevels(in: output))
        XCTAssertEqual(levels.count, 1)
        XCTAssertGreaterThan(levels[0], 0.10)
    }

    func testThreeChannelMicrophoneArrayConvertsToMonoUsingItsStrongestChannel() throws {
        let layout = try XCTUnwrap(
            AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_DiscreteInOrder | 3)
        )
        let inputFormat = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 48_000,
                interleaved: false,
                channelLayout: layout
            )
        )
        let input = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: 4_800))
        input.frameLength = 4_800
        let inputChannels = try XCTUnwrap(input.floatChannelData)
        for frame in 0..<Int(input.frameLength) {
            let sample = Float(0.25 * sin(2 * Double.pi * 440 * Double(frame) / 48_000))
            inputChannels[0][frame] = sample * 0.6
            inputChannels[1][frame] = -sample * 0.6
            inputChannels[2][frame] = sample
        }

        let outputFormat = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16_000,
                channels: 1,
                interleaved: false
            )
        )
        let converter = try XCTUnwrap(AVAudioConverter(from: inputFormat, to: outputFormat))
        converter.downmix = false
        converter.channelMap = [NSNumber(value: 2)]
        let output = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 1_632)
        )
        var suppliedInput = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
            guard !suppliedInput else {
                inputStatus.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            inputStatus.pointee = .haveData
            return input
        }

        XCTAssertNotEqual(status, .error, conversionError?.localizedDescription ?? "conversion failed")
        let levels = try XCTUnwrap(AudioInputLevelMeter.channelRMSLevels(in: output))
        XCTAssertEqual(levels.count, 1)
        XCTAssertGreaterThan(levels[0], 0.15)
    }

    func testInt16InputIsNormalizedForTheMeter() throws {
        let buffer = try makeBuffer(commonFormat: .pcmFormatInt16, channels: 1, frames: 2)
        let samples = try XCTUnwrap(buffer.int16ChannelData)[0]
        samples[0] = 16_384
        samples[1] = -16_384

        let levels = try XCTUnwrap(AudioInputLevelMeter.channelRMSLevels(in: buffer))
        let level = try XCTUnwrap(levels.first)
        XCTAssertEqual(level, 0.5, accuracy: 0.0001)
    }

    func testInt32InputIsNormalizedForTheMeter() throws {
        let buffer = try makeBuffer(commonFormat: .pcmFormatInt32, channels: 1, frames: 2)
        let samples = try XCTUnwrap(buffer.int32ChannelData)[0]
        samples[0] = 1_073_741_824
        samples[1] = -1_073_741_824

        let levels = try XCTUnwrap(AudioInputLevelMeter.channelRMSLevels(in: buffer))
        let level = try XCTUnwrap(levels.first)
        XCTAssertEqual(level, 0.5, accuracy: 0.0001)
    }

    func testSilenceReportsZeroAndEmptyBuffersAreRejected() throws {
        let silent = try makeFloatBuffer(channels: 1, interleaved: false, frames: 3)
        XCTAssertEqual(AudioInputLevelMeter.channelRMSLevels(in: silent), [0.0])

        let empty = try makeFloatBuffer(channels: 1, interleaved: false, frames: 0)
        XCTAssertNil(AudioInputLevelMeter.channelRMSLevels(in: empty))
    }

    private func makeFloatBuffer(
        channels: AVAudioChannelCount,
        interleaved: Bool,
        frames: AVAudioFrameCount,
        sampleRate: Double = 16_000
    ) throws -> AVAudioPCMBuffer {
        let format = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: channels,
                interleaved: interleaved
            )
        )
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(frames, 1))
        )
        buffer.frameLength = frames
        return buffer
    }

    private func makeBuffer(
        commonFormat: AVAudioCommonFormat,
        channels: AVAudioChannelCount,
        frames: AVAudioFrameCount
    ) throws -> AVAudioPCMBuffer {
        let format = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: commonFormat,
                sampleRate: 16_000,
                channels: channels,
                interleaved: false
            )
        )
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(frames, 1))
        )
        buffer.frameLength = frames
        return buffer
    }
}
