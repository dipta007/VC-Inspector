def preprocess_ref(ref):
    caption = []
    for ann in ref["annotations"]:
        vid = ann["image_id"]
        ts = ann["segments"]
        duration = ann["duration"]
        cap_key = "pure_cap" if "pure_cap" in ann else "caption"
        cap = [f"{x.strip()}." for x in ann[cap_key].split(".")]
        caption.append(
            {
                "vid": vid,
                "duration": duration,
                "segments": [{"timestamp": x, "caption": y} for x, y in zip(ts, cap)],
            }
        )
    return caption
