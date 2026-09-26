from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from face_valence import streaming_valence
from pipeline import ConversationPipeline
from transcriber import AudioTranscriber


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
clients: set[WebSocket] = set()
history: deque[dict[str, Any]] = deque(maxlen=120)
transcriber = AudioTranscriber()
conversation = ConversationPipeline()
logger = logging.getLogger("subtext.diagnostics")
ingest_counts: Counter[str] = Counter()
ingest_last_seen: dict[str, float] = {}
ingest_connections = 0
ui_connections = 0


def _record_ingest(name: str) -> None:
    ingest_counts[name] += 1
    ingest_last_seen[name] = time.monotonic()


def _age_since_ingest(name: str) -> str:
    seen = ingest_last_seen.get(name)
    return "never" if seen is None else f"{time.monotonic() - seen:.1f}s"


def _age_since_epoch(timestamp: float | None) -> str:
    return "never" if timestamp is None else f"{max(0.0, time.time() - timestamp):.1f}s"


async def _diagnostic_heartbeat() -> None:
    while True:
        await asyncio.sleep(10)
        audio = transcriber.diagnostics_snapshot()
        face = streaming_valence.diagnostics_snapshot()
        logger.info(
            "DIAG heartbeat | sockets ingest=%d ui=%d | packets mic=%d call=%d visual=%d crops=%d "
            "last mic=%s call=%s visual=%s rejected=%s | audio state=%s model=%s worker=%s "
            "queue=%d/%d dropped=%d worker_reject=%d empty_packets=%d packets_enqueued=%s dequeued=%d "
            "voice_frames=%s quiet_frames=%s starts=%s short=%d rms=%s "
            "transcriptions=%d/%d empty=%d errors=%d in_flight=%s last_whisper_s=%s last_whisper_error=%s "
            "| face state=%s worker=%s queue=%d/%d frames=%d/%d dropped=%d skipped_error=%d "
            "last_in=%s last_done=%s last_crops=%d last_scores=%d last_inference_s=%s errors=%d last_error=%s",
            ingest_connections,
            ui_connections,
            ingest_counts["microphone_chunk"],
            ingest_counts["audio_chunk"],
            ingest_counts["visual_frame"],
            ingest_counts["face_crops"],
            _age_since_ingest("microphone_chunk"),
            _age_since_ingest("audio_chunk"),
            _age_since_ingest("visual_frame"),
            dict((key, value) for key, value in ingest_counts.items() if key.startswith("rejected_") or key.startswith("ignored_")),
            audio["state"],
            audio["model_loaded"],
            audio["worker_alive"],
            audio["queue_size"],
            audio["queue_capacity"],
            audio["queue_dropped_packets"],
            audio["audio_packets_rejected_worker_unavailable"],
            audio["empty_audio_packets"],
            audio["audio_packets_enqueued"],
            audio["audio_packets_dequeued"],
            audio["speech_frames"],
            audio["quiet_frames"],
            audio["utterances_started"],
            audio["utterances_too_short"],
            audio["last_rms"],
            audio["transcriptions_completed"],
            audio["transcriptions_started"],
            audio["transcriptions_empty"],
            audio["transcription_errors"],
            audio["transcription_in_flight"],
            audio["last_transcription_duration_s"],
            audio["last_transcription_error"],
            face["state"],
            face["worker_alive"],
            face["queue_size"],
            face["queue_capacity"],
            face["frames_processed"],
            face["frames_submitted"],
            face["frames_dropped_queue_full"],
            face["frames_skipped_model_error"],
            _age_since_epoch(face["last_input_at"]),
            _age_since_epoch(face["last_processed_at"]),
            face["last_crop_count"],
            face["last_scores_count"],
            face["last_inference_duration_s"],
            face["inference_errors"],
            face["last_error"],
        )


async def broadcast(event: dict[str, Any]) -> None:
    if event.get("type") in {"transcript", "llm_analysis"}:
        history.append(event)

    stale: list[WebSocket] = []
    for client in tuple(clients):
        try:
            await client.send_json(event)
        except Exception as exc:
            logger.warning(
                "Overlay event delivery failed; removing client; error=%s",
                exc,
            )
            stale.append(client)
    for client in stale:
        clients.discard(client)


async def handle_transcriber_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    try:
        if event_type == "audio_cue":
            conversation.add_audio_event(event)
            return
        if event_type == "transcript":
            logger.info(
                "Routing transcript; source=%s chars=%d ui_clients=%d",
                event.get("speaker_id"),
                len(str(event.get("text", ""))),
                ui_connections,
            )
            conversation.add_transcript_event(event)
            await broadcast(event)
            await conversation.maybe_analyze(broadcast)
            return
        await broadcast(event)
    except Exception:
        logger.exception("Transcriber event routing failed; event_type=%s", event_type)
        raise


@asynccontextmanager
async def lifespan(_: FastAPI):
    heartbeat_task = asyncio.create_task(_diagnostic_heartbeat(), name="subtext-diagnostic-heartbeat")
    logger.info("Diagnostic logging enabled; summaries every 10 seconds")
    yield
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    await transcriber.stop()
    streaming_valence.close()


app = FastAPI(title="Subtext local conversation service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "state": transcriber.state,
        "model_loaded": transcriber.model_loaded,
        "face_valence_state": streaming_valence.status,
        "face_valence_detail": streaming_valence.detail,
        "gemini_configured": conversation.interpreter.configured,
        "cue_counts": conversation.cue_counts,
    }


@app.post("/start")
async def start_transcription() -> dict[str, Any]:
    if transcriber.state in {"loading_model", "listening", "transcribing"}:
        return {
            "state": transcriber.state,
            "session_id": conversation.session_id,
            "session_start_epoch": conversation.started_at_epoch,
        }

    start_epoch = time.time()
    conversation.begin(start_epoch)
    logger.info("Starting capture session; session_id=%s", conversation.session_id)
    streaming_valence.reset()
    history.clear()
    await broadcast({"type": "cleared"})
    transcriber.session_start_epoch = start_epoch
    try:
        await transcriber.start(handle_transcriber_event, start_epoch)
    except Exception as exc:
        conversation.clear()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "state": transcriber.state,
        "session_id": conversation.session_id,
        "session_start_epoch": conversation.started_at_epoch,
    }


@app.post("/stop")
async def stop_transcription() -> dict[str, str]:
    logger.info("Stopping capture session; session_id=%s", conversation.session_id)
    await transcriber.stop()
    streaming_valence.reset()
    await broadcast({"type": "valence_cleared"})
    return {"state": transcriber.state}


@app.delete("/transcript")
async def clear_transcript() -> dict[str, str]:
    logger.info("Clearing capture and transcript context")
    history.clear()
    transcriber.clear_context()
    conversation.clear()
    streaming_valence.reset()
    await broadcast({"type": "cleared"})
    return {"state": "cleared"}


@app.websocket("/ws")
async def transcript_socket(websocket: WebSocket) -> None:
    global ui_connections
    await websocket.accept()
    clients.add(websocket)
    ui_connections += 1
    logger.info("Overlay event socket connected; active=%d", ui_connections)
    try:
        await websocket.send_json(
            {
                "type": "status",
                "state": transcriber.state,
                "gemini_configured": conversation.interpreter.configured,
            }
        )
        for event in tuple(history):
            await websocket.send_json(event)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect as exc:
        logger.info("Overlay event socket disconnected; code=%s", exc.code)
    finally:
        clients.discard(websocket)
        ui_connections = max(0, ui_connections - 1)


@app.websocket("/ingest")
async def capture_socket(websocket: WebSocket) -> None:
    """Receive local audio packets, visual measurements, and cropped face frames."""
    global ingest_connections
    await websocket.accept()
    ingest_connections += 1
    logger.info("Capture ingest socket connected; active=%d", ingest_connections)
    try:
        while True:
            raw_message = await websocket.receive_text()
            if len(raw_message) > 180_000:
                _record_ingest("rejected_oversized")
                logger.warning("Rejected oversized capture packet; bytes=%d", len(raw_message))
                await websocket.send_json({"type": "capture_error", "detail": "Capture packet too large."})
                continue
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                _record_ingest("rejected_invalid_json")
                logger.warning("Rejected malformed capture JSON; bytes=%d", len(raw_message))
                continue

            message_type = message.get("type")
            _record_ingest(str(message_type or "unknown_message"))
            if message_type == "microphone_gate":
                enabled = message.get("enabled")
                if isinstance(enabled, bool):
                    transcriber.set_microphone_enabled(enabled)
                else:
                    _record_ingest("rejected_microphone_gate")
                continue
            if message_type == "visual_frame":
                subjects = message.get("subjects", [])
                crop_count = sum(
                    isinstance(subject.get("face_crop_jpeg"), str)
                    for subject in subjects
                    if isinstance(subject, dict)
                )
                ingest_counts["face_crops"] += crop_count
                ingest_counts["faces_detected"] += sum(
                    isinstance(subject, dict) for subject in subjects
                )
                numeric_message = {
                    **message,
                    "subjects": [
                        {
                            key: value
                            for key, value in subject.items()
                            if key not in {"face_crop_jpeg", "person_name"}
                        }
                        for subject in subjects
                        if isinstance(subject, dict)
                    ],
                }
                try:
                    conversation.add_visual_frame(numeric_message)
                except (TypeError, ValueError, ValidationError) as exc:
                    # A malformed visual sample should not tear down the shared
                    # audio/video ingest socket for the rest of the session.
                    _record_ingest("rejected_visual_frame")
                    logger.warning(
                        "Rejected invalid visual frame; error_type=%s",
                        type(exc).__name__,
                    )
                    continue
                if conversation.session_id:
                    streaming_valence.submit(
                        message,
                        session_id=conversation.session_id,
                        publish=broadcast,
                    )
                else:
                    _record_ingest("ignored_visual_no_session")
                await conversation.maybe_analyze(broadcast)
                continue

            if message_type not in {"audio_chunk", "microphone_chunk"}:
                _record_ingest("ignored_unknown_type")
                continue
            if transcriber.state == "idle":
                _record_ingest("ignored_audio_idle")
                continue

            try:
                sample_rate = int(message.get("sample_rate", 0))
                pcm_format = message.get("pcm_format")
                if sample_rate != transcriber.sample_rate or pcm_format not in {"float32", "int16"}:
                    _record_ingest("rejected_audio_format")
                    continue
                encoded = message.get("samples", "")
                if not isinstance(encoded, str) or len(encoded) > 160_000:
                    _record_ingest("rejected_audio_payload")
                    continue
                pcm = base64.b64decode(encoded, validate=True)
                if pcm_format == "float32":
                    if len(pcm) % 4:
                        _record_ingest("rejected_float32_alignment")
                        continue
                    import numpy as np

                    samples = np.frombuffer(pcm, dtype="<f4")
                else:
                    if len(pcm) % 2:
                        _record_ingest("rejected_int16_alignment")
                        continue
                    import numpy as np

                    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                timestamp = float(message.get("timestamp", time.time()))
            except (ValueError, TypeError, binascii.Error):
                _record_ingest("rejected_audio_decode")
                continue

            if message_type == "microphone_chunk":
                _record_ingest("accepted_microphone_chunk")
                transcriber.enqueue_microphone_audio(samples, timestamp)
            else:
                _record_ingest("accepted_audio_chunk")
                transcriber.enqueue_call_audio(samples, timestamp)
    except WebSocketDisconnect as exc:
        logger.warning("Capture ingest socket disconnected; code=%s", exc.code)
    except Exception:
        logger.exception("Capture ingest handler crashed")
        raise
    finally:
        ingest_connections = max(0, ingest_connections - 1)
        logger.info("Capture ingest socket closed; active=%d", ingest_connections)


@app.post("/diagnostics")
async def client_diagnostic(message: dict[str, Any]) -> dict[str, bool]:
    """Print bounded native-app diagnostics in the service terminal."""
    component = str(message.get("component", "unknown"))[:80]
    event = str(message.get("event", "unknown"))[:80]
    detail = str(message.get("detail", ""))[:800]
    metadata = {
        key: value
        for key, value in message.items()
        if key not in {"component", "event", "detail"}
    }
    if event == "capture_heartbeat":
        logger.info(
            "CLIENT diagnostic | component=%s event=%s detail=%s metadata=%s",
            component,
            event,
            detail,
            metadata,
        )
    else:
        logger.warning(
            "CLIENT diagnostic | component=%s event=%s detail=%s metadata=%s",
            component,
            event,
            detail,
            metadata,
        )
    return {"ok": True}
