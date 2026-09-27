import AVFoundation
import Foundation

public enum AudioInputLevelMeter {
    /// Returns normalized RMS for each channel without averaging channels
    /// together. Keeping per-channel levels avoids phase cancellation in
    /// multi-microphone arrays and supports common integer PCM inputs.
    public static func channelRMSLevels(in buffer: AVAudioPCMBuffer) -> [Double]? {
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        guard frameCount > 0, channelCount > 0 else { return nil }
        var sums = [Double](repeating: 0, count: channelCount)

        func addSample(_ rawValue: Double, channel: Int) {
            if rawValue.isFinite {
                sums[channel] += rawValue * rawValue
            }
        }

        switch buffer.format.commonFormat {
        case .pcmFormatFloat32:
            guard let channels = buffer.floatChannelData else { return nil }
            if buffer.format.isInterleaved {
                for frame in 0..<frameCount {
                    for channel in 0..<channelCount {
                        addSample(Double(channels[0][frame * channelCount + channel]), channel: channel)
                    }
                }
            } else {
                for channel in 0..<channelCount {
                    for frame in 0..<frameCount {
                        addSample(Double(channels[channel][frame]), channel: channel)
                    }
                }
            }
        case .pcmFormatInt16:
            guard let channels = buffer.int16ChannelData else { return nil }
            if buffer.format.isInterleaved {
                for frame in 0..<frameCount {
                    for channel in 0..<channelCount {
                        let sample = Double(channels[0][frame * channelCount + channel]) / 32_768.0
                        addSample(sample, channel: channel)
                    }
                }
            } else {
                for channel in 0..<channelCount {
                    for frame in 0..<frameCount {
                        addSample(Double(channels[channel][frame]) / 32_768.0, channel: channel)
                    }
                }
            }
        case .pcmFormatInt32:
            guard let channels = buffer.int32ChannelData else { return nil }
            if buffer.format.isInterleaved {
                for frame in 0..<frameCount {
                    for channel in 0..<channelCount {
                        let sample = Double(channels[0][frame * channelCount + channel]) / 2_147_483_648.0
                        addSample(sample, channel: channel)
                    }
                }
            } else {
                for channel in 0..<channelCount {
                    for frame in 0..<frameCount {
                        addSample(Double(channels[channel][frame]) / 2_147_483_648.0, channel: channel)
                    }
                }
            }
        default:
            return nil
        }

        return sums.map { sqrt($0 / Double(frameCount)) }
    }
}
