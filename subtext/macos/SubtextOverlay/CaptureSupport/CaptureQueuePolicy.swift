import Foundation

/// Chooses which queued packet to discard when the websocket sender falls
/// behind. Local microphone packets are protected from call audio and video.
public enum CaptureQueuePolicy {
    public static func dropIndex(
        incomingMessageType: String,
        queuedMessageTypes: [String]
    ) -> Int? {
        let dropOrder: [String]
        switch incomingMessageType {
        case "microphone_gate", "microphone_chunk":
            dropOrder = ["audio_chunk", "visual_frame", "microphone_chunk"]
        case "audio_chunk":
            // Keep the call's audio timeline intact while there are stale
            // visual updates to evict. Under pressure, visuals are expendable;
            // losing audio packets can make an otherwise clear utterance
            // impossible to transcribe.
            dropOrder = ["visual_frame", "audio_chunk"]
        case "visual_frame":
            dropOrder = ["visual_frame"]
        default:
            dropOrder = ["visual_frame", "audio_chunk"]
        }

        for messageType in dropOrder {
            if let index = queuedMessageTypes.firstIndex(of: messageType) {
                return index
            }
        }
        return nil
    }
}
