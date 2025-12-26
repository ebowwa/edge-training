/*
 * Meta Wearables Mac Receiver
 * Created by humanwritten
 *
 * macOS app entry point for receiving video stream from iOS
 */

import SwiftUI

@main
struct MacReceiverApp: App {
    var body: some Scene {
        WindowGroup {
            MacReceiverContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 800, height: 600)
    }
}
