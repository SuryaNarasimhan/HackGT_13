// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SubtextOverlay",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "SubtextOverlay", targets: ["SubtextOverlay"])
    ],
    targets: [
        .executableTarget(
            name: "SubtextOverlay",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreAudio")
            ]
        )
    ]
)
