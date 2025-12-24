# Meta Wearables Mac Receiver

A macOS companion app that receives video streams from the modified iOS CameraAccess app connected to Meta AI glasses.

## Features

- **Live Video Display**: Shows real-time video stream from Meta glasses
- **MultipeerConnectivity**: Works over WiFi and USB (lowest latency)
- **Auto-Discovery**: Automatically finds iOS devices on the same network
- **Stats Overlay**: Shows FPS, bandwidth, and latency
- **Screenshot**: Save current frame as PNG
- **Recording**: (Coming soon) Record video to file

## Requirements

- macOS 13.0+
- iOS device running the modified CameraAccess app
- Same WiFi network (or USB connection for lowest latency)

## Building

### Using Swift Package Manager

```bash
cd MacReceiver
swift build
swift run
```

### Using Xcode

1. Open `Package.swift` in Xcode
2. Select "My Mac" as the run destination
3. Press Cmd+R to build and run

## Usage

1. Start the Mac Receiver app
2. On your iOS device, open the CameraAccess app
3. Enable "Mac Relay" toggle before starting the stream
4. Start streaming on iOS
5. Video will appear on Mac automatically

## Connection Methods

### WiFi (Easiest)
- Both devices on same WiFi network
- Typical latency: 30-50ms

### USB (Lowest Latency)
- Connect iPhone to Mac via USB/Lightning cable
- MultipeerConnectivity will automatically use the faster USB path
- Typical latency: 10-20ms

## Architecture

```
┌─────────────────┐     Bluetooth     ┌─────────────────┐
│   Meta Glasses  │ ───────────────── │   iOS Device    │
│   (Camera)      │                   │  (CameraAccess) │
└─────────────────┘                   └────────┬────────┘
                                               │
                                     MultipeerConnectivity
                                       (WiFi or USB)
                                               │
                                      ┌────────▼────────┐
                                      │   Mac Receiver  │
                                      │   (This App)    │
                                      └─────────────────┘
```

## Troubleshooting

### Mac not finding iOS device
- Ensure both devices are on the same WiFi network
- Check that "Mac Relay" is enabled on iOS before streaming
- Try connecting via USB cable

### High latency
- Use USB connection instead of WiFi
- Reduce quality setting on iOS app
- Move devices closer to WiFi router

### Dropped frames
- Lower the compression quality on iOS
- Reduce WiFi interference
- Use USB connection

## License

See LICENSE file in the root directory.
