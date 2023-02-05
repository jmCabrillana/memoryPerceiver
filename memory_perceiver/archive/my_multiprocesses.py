from pytorchvideo.data.encoded_video import EncodedVideo

def slice_medias(items):
    sliced_medias=[]
    for path, start, stop in items:
        try:
            media = EncodedVideo.from_path(path)
            sliced_media = media.get_clip(start_sec=start, end_sec=stop)
            sliced_medias.append(sliced_media)
        except:
            print('error with item: ', media, start, stop)
    return sliced_medias