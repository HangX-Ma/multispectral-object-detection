from sahi.predict import get_prediction, get_sliced_prediction, predict

model_type = "cft"
model_path = 'best.pt'
model_device = "0" # or 'cpu'
model_confidence_threshold = 0.8

slice_height = 512
slice_width = 512
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "/hy-tmp/LLVIP/visible/images"

predict(
    model_type=model_type,
    model_path=model_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)