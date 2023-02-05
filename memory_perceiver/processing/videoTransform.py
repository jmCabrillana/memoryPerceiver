from torchvision.transforms import Compose, Lambda, Resize, CenterCrop, Normalize
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

# Define transformation

transform_video =  lambda nb_frames : ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(nb_frames),
            Lambda(lambda x: x/255.0),
            ShortSideScale(size=256),
            CenterCrop(
                (256, 456)
            )
        ]
    ),
)

transform_audio = lambda nb_frames : ApplyTransformToKey(
    key="audio",
    transform=Compose(
        [
            UniformTemporalSubsample(nb_frames*1920, temporal_dim=0),
        ]
    ),
)