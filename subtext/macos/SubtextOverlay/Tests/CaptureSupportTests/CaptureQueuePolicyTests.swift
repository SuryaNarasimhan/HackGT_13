import XCTest
@testable import CaptureSupport

final class CaptureQueuePolicyTests: XCTestCase {
    func testMicrophonePacketEvictsCallAudioBeforeQueuedMicrophone() {
        let queued = ["microphone_chunk", "visual_frame", "audio_chunk", "microphone_chunk"]

        XCTAssertEqual(
            CaptureQueuePolicy.dropIndex(
                incomingMessageType: "microphone_chunk",
                queuedMessageTypes: queued
            ),
            2
        )
    }

    func testCallAudioNeverEvictsQueuedMicrophone() {
        let queued = ["microphone_chunk", "microphone_chunk", "visual_frame"]

        XCTAssertEqual(
            CaptureQueuePolicy.dropIndex(
                incomingMessageType: "audio_chunk",
                queuedMessageTypes: queued
            ),
            2
        )
    }

    func testCallAudioEvictsVisualFramesBeforeDiscardingCallAudio() {
        let queued = ["audio_chunk", "microphone_chunk", "visual_frame", "audio_chunk"]

        XCTAssertEqual(
            CaptureQueuePolicy.dropIndex(
                incomingMessageType: "audio_chunk",
                queuedMessageTypes: queued
            ),
            2
        )
    }

    func testMicrophoneQueueIsTheLastResortWhenQueueContainsOnlyMicrophone() {
        let queued = ["microphone_chunk", "microphone_chunk"]

        XCTAssertEqual(
            CaptureQueuePolicy.dropIndex(
                incomingMessageType: "microphone_chunk",
                queuedMessageTypes: queued
            ),
            0
        )
    }
}
