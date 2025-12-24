// swift-tools-version: 5.9
// Meta Wearables Mac Receiver

import PackageDescription

let package = Package(
    name: "MacReceiver",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MacReceiver", targets: ["MacReceiver"])
    ],
    targets: [
        .executableTarget(
            name: "MacReceiver",
            path: "MacReceiver"
        )
    ]
)
