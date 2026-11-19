# Face Censor - OBS plugin

Video filter for webcams: detects faces and covers them with blur, a black box, TV static or pixelation.

## Get the plugin (GitHub builds it for you, no programming tools needed)

1. Make a free account at github.com, click **New repository**, name it `obs-face-censor`, keep it Public (free build minutes), create it.
2. On the new repo page click **uploading an existing file**. Unzip this project and drag **everything inside the folder**
   (including the hidden `.github` folder, `src`, `data`, `cmake`, `CMakeLists.txt`...) into the page, then **Commit changes**.
   If `.github` doesn't upload by dragging, use GitHub Desktop (desktop.github.com): add the folder as a repository and Publish.
3. Open the **Actions** tab. A run called "Build Windows plugin" starts automatically (about 5-10 minutes).
4. When it shows a green tick, open it and download the **obs-face-censor-windows** artifact at the bottom. Unzip it.
5. Copy the `obs-face-censor` folder into `C:\ProgramData\obs-studio\plugins\`
   (so you have `...\plugins\obs-face-censor\bin\64bit\obs-face-censor.dll`). Restart OBS.

If the run fails (red cross), open it, copy the error text from the failed step and send it to me.

## Use it

Right-click your webcam source > **Filters** > **+** > **Face Censor (blur / black box / static)**.
Pick the style, padding, and how long to keep censoring after a face is lost.

## Notes

- Face model included: `data/models/face_detection_yunet_2023mar.onnx` (YuNet, OpenCV Zoo).
- If the model can't load, the filter blacks out the whole frame by default (setting: fail closed).
- Detection runs ~15x/second. Very fast head turns can briefly outrun the box: raise Padding.
- Face detection can miss extreme angles, heavy occlusion or very low light.
