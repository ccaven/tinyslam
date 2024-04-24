# tiny_slam

`**tiny_slam**` is a visual SLAM (Simultaneous Localization and Mapping) library. It heavily relies on general-purpose GPU computing via the `**wgpu**` library (primarily using the Vulkan backend).

This library is a work in progess.

## Motivation

`**tiny_slam**` aims to:

1. Make visual SLAM accessible to developers, independent researchers, and small companies
2. Decrease the cost of visual SLAM
3. Bring edge computing to cross-platform devices (via `**wgpu**`)
4. Increase innovation in drone / autonomous agent applications that are unlocked given precise localization

## Constraints

`**tiny_slam**` imposes these constraints on itself:

1. Minimize number of dependencies
2. Rely on computer shaders whenever possible
3. Run in realtime on a Raspberry Pi 5
4. Ergonomic design (Rust-like)

## Progess

- [ ]  Obtain required hardware
    - [x]  Raspberry Pi 5
    - [ ]  High framerate camera, like [this one](https://www.amazon.com/Arducam-Distortion-Microphones-Computer-Raspberry/dp/B096M5DKY6?ref_=ast_sto_dp&th=1&psc=1)
    - [ ]  Drone materials
- [ ]  Receive realtime data from USB camera
    - [x]  Use MediaFoundation API on Windows, video4linux on Linux, and AVFoundation on MacOS
    - [x]  Software decode Motion JPEG (MJPEG) stream from webcam
    - [ ]  Utilize hardware decoding with higher framerate cameras
- [x]  Build helper library (`**tiny_wgpu**`) to increase compute shader workflow
    - [x]  Increase default limits for push constants and number of bindings
    - [x]  Enable read/write storage textures
    - [x]  Support render pipelines
    - [x]  Support reading data back to CPU via staging buffers
    - [x]  Support multiple shader files
- [ ]  Feature detection
    - [ ]  Color to grayscale conversion
        - [x]  Implement manual luminance calculation
        - [ ]  (Optional) Use Y channel of YUV stream directly
    - [x]  Oriented FAST corner detection
        - [x]  Implement workgroup optimizations
        - [x]  Implement bitwise corner detector
        - [x]  Implement 4-corner shortcut
        - [x]  Replace storage buffers with textures to improve memory reads
    - [x]  Rotated BRIEF feature descriptors
        - [x]  Two-pass gaussian blur
        - [x]  Use linear sampler filtering to decrease number of samples
        - [ ]  Implement workgroup optimizations
    - [ ]  Read data back to CPU
- [ ]  Local mapping
    - [ ]  Keyframe selection
    - [ ]  Insertion into current Map
    - [ ]  Cull unnecessary map points
    - [ ]  Local bundle adjustment