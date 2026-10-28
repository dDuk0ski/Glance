# Glance

An AI assistant that lives in the MacBook notch.

CSI-4130/5130 Artificial Intelligence, Course Project (Fall 2026)

**Authors:** Damjan Dukoski, Spencer Roeren, Michael Medulla

## Problem

Today's AI assistants live in separate windows and tabs. To use them, you have to switch apps, copy and paste, and re-explain what is already on your screen.

## Proposal

Glance is a macOS assistant that lives in the MacBook notch: press a shortcut, type a prompt into a Liquid Glass input, and the LLM's reply streams out of the notch without you leaving the app you're in. Glance closes the context gap with an eye toggle that attaches a screenshot to the prompt, and a file shelf whose items can each be included in or excluded from what the model sees. We build on the open-source boring.notch app (GPL-3.0), removing its media features and adding a model-agnostic layer through OpenRouter so the same interface can run several state-of-the-art models. To keep replies on screen only while they're needed, Glance uses on-device head-pose estimation from the webcam (Apple Vision) to tell whether the user is looking at the notch, and dismisses the reply when they look away.

## Method

- **Notch UI:** SwiftUI app forked from boring.notch; notch window, hover/gesture handling and file shelf are reused.
- **LLM layer:** streaming chat completions through OpenRouter (multiple providers and models behind one API).
- **Screen context:** ScreenCaptureKit screenshot, attached on demand, with Glance's own window excluded.
- **File context:** shelf items can each be toggled in or out of the prompt; a meter shows how much of the model's context they use.
- **Attention detection:** Apple Vision face and head-pose features from the webcam, compared against a small trained classifier.

## Data and Evaluation

1. **Attention dataset:** labeled webcam frames of users looking at the notch versus elsewhere. We will measure accuracy, latency and false-dismissal rate.
2. **Screen-QA benchmark:** desktop screenshots paired with questions and reference answers. We will compare models, image resolutions, and screenshot versus extracted-text (OCR) inputs on accuracy, latency and cost.

## Building

Requires macOS 14+ and Xcode 16+.

```bash
open Glance.xcodeproj
```

Build and run the `Glance` scheme. Swift packages resolve automatically on first build.

## Citations and Acknowledgements

- **[boring.notch](https://github.com/TheBoredTeam/boring.notch)** by TheBoredTeam (GPL-3.0). Glance is a fork of boring.notch at commit `23e7705`, with its media, calendar, battery, HUD, download, onboarding and auto-update features removed. The notch window, shelf and settings code originate from it.
- Libraries: [Defaults](https://github.com/sindresorhus/Defaults), [KeyboardShortcuts](https://github.com/sindresorhus/KeyboardShortcuts), [LaunchAtLogin-Modern](https://github.com/sindresorhus/LaunchAtLogin-Modern) by Sindre Sorhus (MIT).
- Further third-party notices are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

## License

GPL-3.0, inherited from boring.notch. See [LICENSE](LICENSE).
