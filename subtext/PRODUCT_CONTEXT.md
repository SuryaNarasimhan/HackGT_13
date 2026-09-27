Subtext — Product Context

Overview

Subtext is an AI-powered communication layer designed to help neurodivergent and neurotypical people understand one another better during live video conversations.

The core idea is to use multimodal signals—such as spoken language, vocal characteristics, facial/visual behavior, and conversation dynamics—to identify moments where communication may be ambiguous or where two people may misunderstand each other.

Subtext is intended to help strengthen human-to-human connection, not replace it.

This project is currently being developed as a hackathon concept, so the exact product behavior, model stack, architecture, and supported cues are not finalized.

Problem

Many conversations depend on information that is never stated explicitly.

Examples include:

indirect requests

implied disagreement

sarcasm or teasing

hesitation

emotional changes

conversational turn-taking

expectations that someone should respond

subtle signs that someone may be confused, uncomfortable, frustrated, excited, or sad

Some neurodivergent people may have difficulty interpreting certain implicit social signals in real time.

At the same time, communication difficulties are not necessarily one-sided.

A neurotypical person may also misunderstand a neurodivergent person's:

direct communication

longer processing pauses

reduced facial expressiveness

different vocal prosody

literal language

different expectations around turn-taking

Subtext should therefore avoid assuming that one communication style is "correct."

Double-Empathy Framing

A major principle behind the product is the double empathy problem.

Instead of:

Neurodivergent person fails to interpret neurotypical communication.

Subtext considers:

Two people with different communication styles may have difficulty interpreting each other.

The product should therefore function as a communication bridge, rather than a tool that teaches neurodivergent users to behave more neurotypically.

The system should analyze interactions, not judge individuals.

Product Vision

A user enables Subtext while having a video call through something like:

FaceTime

Zoom

Google Meet

Discord

Teams

another video communication platform

Subtext operates in the background.

Most of the time, the user sees nothing.

When the system detects a potentially important communication moment, it may show a small temporary cue.

Example:

Possible indirect request
"It would be nice if someone looked over this."
They may be asking for someone to review it.

Or:

Possible frustration
Their responses have become shorter and their vocal intensity increased.

Or:

Meaning unclear
Their words indicate agreement, but other signals are mixed. Consider clarifying instead of assuming.

The goal is to provide context, not definitive psychological conclusions.

Core Design Principle

Subtext should answer questions such as:

What happened?

Example:

They tried to begin speaking twice while you were talking.

What might they mean?

Example:

Their statement may be an indirect suggestion to change plans.

How might they be feeling?

Example:

Several signals may indicate frustration.

Could we be misunderstanding each other?

Example:

Their communication is very direct, but the disagreement appears focused on the idea rather than on you personally.

Should we clarify?

Example:

The meaning is ambiguous. You may want to ask what they meant rather than assume.

Possible Cue Categories

The exact labels are still open to experimentation, but current thinking falls into roughly five groups.

1. Emotional / Affective Cues

Potential states could include:

frustration

sadness / low mood

nervousness

discomfort

confusion

excitement / enthusiasm

These should never be presented as certain facts.

Avoid:

They are angry.

Prefer:

Possible frustration.

Or:

Their vocal and visual behavior changed in ways that may indicate frustration.

2. Implied Meaning / Social Intent

Potential detections:

indirect requests

indirect invitations

soft rejection

indirect disagreement

sarcasm

teasing

hedging

uncertainty

implied concern

Example:

"Maybe we should consider another approach."

could be surfaced as:

Possible indirect disagreement
They may be suggesting that the current approach should change.

3. Conversation Dynamics

These are generally more observable and less inferential.

Examples:

someone is waiting for a response

someone attempted to speak

repeated interruptions

overlapping speech

an unanswered question

a long pause

abrupt topic change

one participant has stopped contributing

possible end-of-conversation signals

Example:

Possible missed turn
Alex attempted to begin speaking twice.

4. Communication-Style Differences

Subtext should account for different communication styles rather than interpreting everything through neurotypical norms.

Examples:

direct vs. indirect communication

literal vs. implied language

different processing times

different levels of facial expressiveness

different vocal prosody

different turn-taking expectations

Example:

Direct communication
Their disagreement appears focused on the idea. Direct wording does not necessarily indicate hostility.

5. Ambiguity / Misunderstanding Risk

Sometimes the most useful output is:

We do not know.

Examples:

verbal and nonverbal signals conflict

several interpretations are plausible

insufficient evidence

behavior differs from expectations but has many possible explanations

Subtext might respond:

Meaning unclear
There isn't enough evidence to infer their intent confidently.

And optionally suggest:

"Just to make sure I understand—what did you mean by that?"

Multimodal Approach

A central idea of Subtext is that social communication is multimodal.

The system may combine:

Language

Potential information:

actual words spoken

sentence meaning

hedging

questions

indirect requests

disagreement

sarcasm candidates

uncertainty

changes in response length

Audio

Potential information:

response latency

pauses

speaking rate

pitch

pitch variation

vocal intensity

volume

filler sounds

sighs

overlap

interruptions

changes relative to the speaker's baseline

Visual Behavior

Potential information:

facial movement

changes in facial behavior

head nodding

head shaking

head orientation

posture

gestures

mouth opening before attempting to speak

changes relative to previous behavior

Subtext should be cautious about directly mapping a facial expression to an emotion.

For example:

Avoid:

Face → angry.

Prefer:

Facial behavior changed + voice intensity increased + responses shortened → possible frustration.

Conversation Context

Potential information:

who spoke previously

whether a question is pending

how long someone has been silent

whether someone repeatedly tried to speak

previous communication patterns

changes relative to earlier parts of the conversation

Context is important because the same behavior can mean different things in different situations.

Possible System Architecture

Nothing here is finalized.

A possible pipeline is:

Live video conversation
        |
        +---- audio
        |
        +---- video
        |
        +---- microphone
        |
        v
Signal processing / pretrained models
        |
        +---- transcription
        +---- language features
        +---- acoustic features
        +---- visual features
        +---- conversation timing
        |
        v
Synchronized conversation windows
        |
        v
Multimodal reasoning / fusion
        |
        v
Potential communication event
        |
        v
Intervention decision
        |
     +--+--+
     |     |
   ignore  surface
            |
            v
     lightweight cue

Model Philosophy

The project may use multiple pretrained models rather than attempting to train everything from scratch.

Possible components include models or tools for:

speech-to-text

voice activity detection

language embeddings

speech emotion / acoustic embeddings

facial landmarks

pose estimation

conversation-state tracking

A smaller custom model or reasoning layer could combine these signals.

The exact technologies are still open.

Current candidates that have been discussed include tools such as:

Whisper / whisper.cpp

Silero VAD

MediaPipe

emotion-oriented speech embeddings

sentence embeddings

lightweight PyTorch fusion models

LLMs for explanation

These should be treated as possible implementation choices, not commitments.

Role of an LLM

One possible architecture separates:

Detection

Models determine:

Something potentially important happened.

from:

Explanation

An LLM converts structured evidence into concise language.

Example input:

event: possible indirect disagreement

evidence:
- qualified agreement
- alternative proposed
- long pause before response

Possible output:

Possible disagreement
They agreed initially but then suggested another option, which may indicate reservations.

This keeps an LLM from simply inventing emotional interpretations from raw conversation data.

User Experience

Subtext should be ambient.

The system watches continuously so that the user does not need to.

The interface should avoid a constantly updating dashboard or sidebar that forces users to divide their attention between the conversation and the AI.

Possible interaction:

Normal conversation
        |
        v
nothing displayed

Potential meaningful event
        |
        v
small temporary cue

User ignores it
        |
        v
cue disappears

The user could optionally expand the cue.

Example:

Possible frustration

Their responses became shorter and
their vocal intensity increased.

[Why?]  [Clarify]

Selecting Why? could show the evidence.

Selecting Clarify could generate a simple clarification question.

Clarification Rather Than Mind Reading

A major product principle is that Subtext should encourage communication repair.

Instead of telling the user:

They are annoyed.

it may say:

Their response could have several interpretations.

Then:

You could ask: "Are you actually okay with that?"

The AI should help people communicate with each other, rather than becoming an authority on what another person thinks.

Private vs. Shared Cues

One concept under consideration is distinguishing:

Private accessibility cues

Visible only to the Subtext user.

Examples:

possible indirect request

possible sarcasm

possible emotional change

possible soft rejection

Shared interaction cues

Potentially visible to everyone.

Examples:

two people started speaking simultaneously

a question may still be unresolved

someone has repeatedly attempted to speak

For an initial MVP, private cues may be simpler and safer.

Personalization

Subtext should probably avoid a generic:

Autism mode

or:

Neurodivergent mode.

Instead, users could specify what kinds of communication support they find useful.

Possible preferences:

explain indirect language

flag ambiguous statements

help with sarcasm

notify me when someone is waiting for a response

identify attempted turns

surface possible emotional changes

suggest clarification questions

This recognizes that neurodivergent people are not a homogeneous group.

Platform Direction

The long-term idea is for Subtext to work as an accessibility layer over existing communication platforms, rather than requiring people to use a new calling service.

Potential platforms include:

FaceTime

Zoom

Meet

Discord

Teams

Current prototyping is beginning on macOS.

One possible implementation is a standalone application that:

observes a selected call window/audio stream

performs analysis locally or through a backend

displays a floating overlay above the call

Cross-platform support is a later consideration.

Hackathon MVP Direction

The hackathon MVP does not need to solve every social interaction.

The goal is to demonstrate that:

multimodal AI can meaningfully assist human-to-human communication in real time.

A possible MVP might support only a few events.

For example:

Indirect communication

"Maybe we should try something else."

→ possible indirect disagreement / suggestion

Emotional change

Several audio + visual signals change simultaneously.

→ possible frustration

Turn-taking

Someone repeatedly attempts to speak.

→ possible missed conversational turn

The exact events remain open to experimentation.

Suggested MVP Development Order

A possible sequence is:

Floating overlay

Live transcription

Capture both sides of a call

Conversation timing / voice activity

Language-intent analysis

Audio-affect features

Visual features

Synchronization

Multimodal fusion

Intervention filtering

Natural-language explanation

Demo polish

The guiding principle is to maintain a working product at each stage rather than attempting the entire multimodal system at once.

Differentiation

There are already products for:

meeting transcription

sales coaching

speaker coaching

emotion recognition

text tone analysis

neurodivergent communication guidance

Subtext should not compete primarily on those dimensions.

Its proposed differentiation is:

real-time multimodal support for mutual understanding between people with different communication styles.

Most communication AI asks:

How can this person communicate more effectively?

Subtext asks:

Where might these two people be misunderstanding each other, and how can we help them reconnect?

This distinction should remain central.

What Subtext Should Not Become

Avoid drifting into:

Sales coaching

Examples to avoid:

persuasion score

charisma score

speaking-performance score

Behavioral normalization

Examples:

"Make more eye contact."

"Smile more."

"Your communication is abnormal."

Definitive emotion recognition

Examples:

"They are angry."

"They are lying."

"They don't like you."

Constant interruption

The product should reduce cognitive load, not create another stream of information that users must constantly monitor.

Meta Track Fit

Challenge direction:

Build an AI-powered social product that helps strengthen connections in new ways.

Subtext's central argument is:

Most social AI creates content, recommends content, or replaces parts of human interaction. Subtext uses AI to improve understanding inside existing human relationships.

The AI is therefore positioned between people to facilitate connection, rather than as the connection itself.

ML Track Fit

The interesting ML problem is not simply:

classify facial emotions.

Instead:

combine language, acoustic behavior, visual behavior, timing, and conversation context to detect potentially meaningful communication events.

Potential technical themes include:

multimodal feature fusion

temporal modeling

uncertainty estimation

personalized baselines

communication-event classification

intervention thresholding

explainability

The system should ideally demonstrate that multiple modalities together provide useful evidence beyond any single modality.

Important Open Questions

These are intentionally unresolved.

Detection

Which social cues are useful enough to include?

Which cues can realistically be detected reliably?

How should emotional states be represented?

How should ambiguity be represented?

Should the model classify individual states or interaction-level events?

Multimodal ML

Should fusion be rule-based, learned, or hybrid?

Should models operate per utterance or over temporal windows?

How much conversation history is necessary?

What data can be used for training/evaluation?

How should confidence be calibrated?

Personalization

Should the model establish individual behavioral baselines?

How quickly could it learn them?

Which preferences should users control?

UX

How often should Subtext intervene?

Which cues deserve immediate display?

Which cues should only appear on demand?

What is the least distracting presentation?

Should there be an after-call review?

Privacy

Can inference remain entirely local?

Should conversations ever be stored?

How should other participants be informed?

How should consent work?

Platform

How should audio/video be captured reliably?

How should the overlay behave across different calling apps?

What platform should be targeted after macOS?

North Star

Subtext should not attempt to make AI an expert on people's emotions.

The product should instead help answer:

"Is there something happening in this interaction that I might want to notice, understand, or clarify?"

The desired outcome is not better performance.

It is better mutual understanding between people.

## Current prototype implementation

The first end-to-end technical path is now implemented as a prototype:

- The macOS app captures the selected microphone and one selected call window.
- faster-whisper transcribes both audio sources locally; audio features include turn timing, pauses, approximate pitch, loudness, speaking rate, and change from each source's short session baseline.
- Apple Vision analyzes selected-window frames locally at up to two frames per second and extracts face presence, face position, head orientation, and mouth/eye aperture measurements. These are behavior measurements, not emotion labels or identities.
- Transcript, audio, and visual cues use shared session-relative timestamps and are sent as sorted, one-second-aligned records inside recent conversation windows.
- When configured, Gemini receives the transcript and compact numeric cues for interpretation. Raw audio and video stay on the Mac. The response is schema-validated and framed as observations or tentative hypotheses with alternatives and a possible check-in.
- Ambiguous windows should remain quiet in the overlay; a possible cue is brief and expires automatically.

These implementation choices support the north star and remain prototype decisions. They do not settle future product behavior, cue thresholds, privacy/consent policy, or the final model stack.

## Project clarification

Subtext is primarily for personal relationships, such as conversations between partners, friends, and family—not business meetings or workplace performance. Examples, cue priorities, and evaluation should reflect relational moments such as possible hurt, discomfort, affection, or bids for connection. Cues should support a gentle check-in and mutual understanding, not task coordination or productivity.

## Meta Track north star

Subtext should bring people together in a distinctive way by helping them understand and reconnect with each other during existing personal conversations. AI acts as a discreet bridge within the interaction; the people remain the relationship and decide what a moment means. Product success is stronger mutual understanding and connection between participants.
