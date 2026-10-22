
# SignBridge Landmark Contract — Draft v1

## Source
MediaPipe landmark outputs extracted locally in the browser
or by the Python data recorder.

## Per-frame layout
1. Left hand: 21 points × (x, y, z) = 63 values
2. Right hand: 21 points × (x, y, z) = 63 values
3. Pose: 33 points × (x, y, z, visibility) = 132 values

Total: 258 values per frame.

## Ordering
- Hand points follow MediaPipe's documented hand landmark order.
- Pose points follow MediaPipe's documented pose landmark order.
- Left and right hands occupy separate fixed slots.
- Missing hands use zero-filled coordinates and a separate
  presence mask in the preprocessing pipeline.
- Missing pose landmarks use zero-filled coordinates and
  visibility = 0.

## Coordinate conventions
- Preserve MediaPipe coordinates as raw extraction output.
- Do not mix raw coordinates with normalized coordinates.
- Normalization and velocity features will be specified in
  preprocessing-spec.md before model training.
- Training and browser inference must use identical preprocessing.

## Sequence conventions
- Target sequence length: 30 frames initially.
- The model input shape will be [batch, time, features].
- Final feature count and data type will be frozen before
  exporting ONNX.

## Versioning
Changes to ordering, coordinate conventions, or feature count
require a contract version change and a compatible model.
