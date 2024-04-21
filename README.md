# orbslam_gpu

This project aims to implement ORB-SLAM using `wgpu` compute shaders.

Progess on compute shader items:
 - [ ] Hardware accelerated video decoding
 - [x] FAST Corner Detection
 - [ ] Hierarchical features
 - [x] Gaussian blur
 - [x] Rotated BRIEF Feature Descriptors
 - [ ] Feature matching
 - [ ] Keypoint detection

Progress on non-compute shader items:
 - [ ] Local bundle adjustment
 - [ ] Global bundle adjustment / loop closing

## Implementation Notes

This project uses the [tiny_wgpu](https://github.com/ccaven/tiny_wgpu) project to reduce the amount of `wgpu` boilerplate.