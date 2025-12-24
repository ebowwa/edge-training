/*
 * Meta Wearables Mac Receiver
 * Created by humanwritten
 *
 * Receives video stream from iOS app connected to Meta AI glasses
 * via MultipeerConnectivity (works over WiFi and USB)
 */

import SwiftUI

@main
struct MacReceiverApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 800, height: 600)
    }
}
