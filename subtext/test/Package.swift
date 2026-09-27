// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MeetingOverlayTest",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "MeetingOverlayTest", targets: ["MeetingOverlayTest"])
    ],
    targets: [
        .executableTarget(
            name: "MeetingOverlayTest",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("AVFoundation")
            ]
        )
    ]
)
