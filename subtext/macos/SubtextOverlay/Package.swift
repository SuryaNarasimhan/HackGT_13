// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SubtextOverlay",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "SubtextOverlay", targets: ["SubtextOverlay"])
    ],
    targets: [
        .target(
            name: "CaptureSupport",
            path: "CaptureSupport",
            linkerSettings: [
                .linkedFramework("AVFoundation")
            ]
        ),
        .executableTarget(
            name: "SubtextOverlay",
            dependencies: ["CaptureSupport"],
            path: "Sources",
            linkerSettings: [
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreAudio")
            ]
        ),
        .testTarget(
            name: "CaptureSupportTests",
            dependencies: ["CaptureSupport"],
            path: "Tests/CaptureSupportTests"
        )
    ]
)
