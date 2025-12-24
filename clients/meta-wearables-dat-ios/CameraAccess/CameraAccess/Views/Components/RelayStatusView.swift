/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Added by humanwritten for Mac relay functionality
 */

//
// RelayStatusView.swift
//
// UI component showing Mac relay connection status and controls
//

import SwiftUI
import MWDATCore

struct RelayStatusView: View {
    @ObservedObject var viewModel: StreamSessionViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with toggle
            HStack {
                Image(systemName: "desktopcomputer")
                    .foregroundColor(.blue)
                Text("Mac Relay")
                    .font(.headline)

                Spacer()

                Toggle("", isOn: $viewModel.isRelayEnabled)
                    .labelsHidden()
            }

            if viewModel.isRelayEnabled {
                // Connection status
                HStack {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 10, height: 10)
                    Text(statusText)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                // Connected Macs
                if !viewModel.connectedMacs.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Connected:")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        ForEach(viewModel.connectedMacs, id: \.self) { mac in
                            HStack {
                                Image(systemName: "laptopcomputer")
                                    .font(.caption)
                                Text(mac)
                                    .font(.caption)
                            }
                            .foregroundColor(.green)
                        }
                    }
                }

                // Stats when streaming
                if viewModel.relayConnectionState.isConnected && viewModel.isStreaming {
                    HStack(spacing: 16) {
                        // Frame rate
                        VStack(alignment: .leading) {
                            Text("FPS")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.1f", viewModel.relayFrameRate))
                                .font(.caption)
                                .monospacedDigit()
                        }

                        // Bandwidth
                        VStack(alignment: .leading) {
                            Text("Bandwidth")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(formatBandwidth(viewModel.relayBytesPerSecond))
                                .font(.caption)
                                .monospacedDigit()
                        }

                        Spacer()

                        // Quality slider
                        VStack(alignment: .leading) {
                            Text("Quality: \(Int(viewModel.relayCompressionQuality * 100))%")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Slider(value: $viewModel.relayCompressionQuality, in: 0.1...1.0, step: 0.1)
                                .frame(width: 80)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var statusColor: Color {
        switch viewModel.relayConnectionState {
        case .disconnected:
            return .red
        case .connecting:
            return .orange
        case .connected:
            return .green
        }
    }

    private var statusText: String {
        switch viewModel.relayConnectionState {
        case .disconnected:
            return "Waiting for Mac..."
        case .connecting:
            return "Connecting..."
        case .connected(let count):
            return "\(count) Mac\(count == 1 ? "" : "s") connected"
        }
    }

    private func formatBandwidth(_ bytesPerSecond: Int) -> String {
        if bytesPerSecond > 1_000_000 {
            return String(format: "%.1f MB/s", Double(bytesPerSecond) / 1_000_000)
        } else if bytesPerSecond > 1_000 {
            return String(format: "%.0f KB/s", Double(bytesPerSecond) / 1_000)
        } else {
            return "\(bytesPerSecond) B/s"
        }
    }
}

#Preview {
    RelayStatusView(viewModel: StreamSessionViewModel(wearables: Wearables.shared))
        .padding()
}
