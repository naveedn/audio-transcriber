// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ParakeetTranscriber",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(
            name: "parakeet-transcriber",
            targets: ["ParakeetTranscriber"]
        ),
    ],
    dependencies: [
        .package(
            path: "FluidAudio"
        ),
    ],
    targets: [
        .executableTarget(
            name: "ParakeetTranscriber",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/ParakeetTranscriber"
        ),
    ]
)
