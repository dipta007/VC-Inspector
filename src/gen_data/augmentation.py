import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from openai import OpenAI

from .model_engine import VLLM
from .prompts import PROMPT_SET

sen_model = None


def get_similarity(embeddings1, embeddings2):
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)

    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    embeddings1_normalized = embeddings1 / norm1
    embeddings2_normalized = embeddings2 / norm2

    similarities = embeddings1_normalized @ embeddings2_normalized.T
    return similarities


def get_similarity_score(inst, queries, candidates):
    query_texts = [inst.format(query=q) for q in queries]
    query_embeddings = []
    for query_text in query_texts:
        query_embedding = sen_model.embeddings.create(
            input=query_text, model="Qwen3-Embedding-8B", encoding_format="float"
        )
        query_embeddings.append(query_embedding.data[0].embedding)
    candidate_embeddings = []
    for candidate in candidates:
        candidate_embedding = sen_model.embeddings.create(
            input=candidate, model="Qwen3-Embedding-8B", encoding_format="float"
        )
        candidate_embeddings.append(candidate_embedding.data[0].embedding)
    scores = get_similarity(query_embeddings, candidate_embeddings)
    return scores


def get_harmonic_mean(scores):
    up = len(scores)
    down = np.sum(1 / scores)
    return up / down


def get_score(originals, augments, original_cap, augment_cap):
    assert len(originals) == len(augments), (
        f"Length of originals and augments must be the same, but got {len(originals)} and {len(augments)}"
    )
    if len(originals) == 0:
        return None, None, 1

    INST = "Instruct: Given a word or phrase, retrieve the similar words or phrases.\nQuery:{query}"
    scores = get_similarity_score(INST, originals, augments)
    scores = np.diag(scores)

    INST = "Instruct: Given a caption, retrieve the similar captions.\nQuery:{query}"
    similarity = get_similarity_score(INST, [original_cap], [augment_cap])
    similarity = similarity[0][0]

    harmonic_mean = get_harmonic_mean(scores)
    final_score = 0.3 * harmonic_mean + 0.7 * similarity
    return scores.tolist(), similarity, final_score


def change_obj(args, seg, model):
    st, end = seg["timestamp"]
    cap = seg["caption"]
    # parse objects in the caption
    prompt_str = PROMPT_SET["obj_list"].format(cap=cap)
    obj_list = eval(model.prompt_text(prompt_str))
    n_obj = len(obj_list)

    segments = []
    aug_cap_set = set()
    for k in range(args.n_sample):
        cap = seg["caption"]
        np.random.shuffle(obj_list)
        n_obj_replace = np.random.choice(n_obj + 1)
        wrong_objs = []
        originals = []
        augments = []
        for i in range(n_obj_replace):
            obj = obj_list[i]
            prompt_str = PROMPT_SET["new_obj"].format(obj=obj)
            obj_new = model.prompt_text(prompt_str)
            wrong_objs.append(obj_new)
            prompt_str = PROMPT_SET["substitute"].format(ow=obj, nw=obj_new, cap=cap)
            cap = model.prompt_text(prompt_str)
            originals.append(obj.strip())
            augments.append(obj_new.strip())

        if cap in aug_cap_set:
            continue
        aug_cap_set.add(cap)
        quality = round(1 - float(n_obj_replace) / (n_obj + 1), 4)

        scores, similarity, quality = get_score(
            originals, augments, seg["caption"], cap
        )
        explanation = {"wrong objects": wrong_objs}
        segments.append(
            {
                "timestamp": [st, end],
                "caption": cap,
                "quality": quality,
                "explanation": explanation,
                "original_objects": originals,
                "augmented_objects": augments,
                "original_caption": seg["caption"],
                "augmented_caption": cap,
                "obj_vs_obj": scores,
                "caption_vs_caption": similarity,
            }
        )
    return segments


def change_act(args, seg, model):
    st, end = seg["timestamp"]
    cap = seg["caption"]

    # parse actions in the caption
    prompt_str = PROMPT_SET["act_list"].format(cap=cap)
    act_list = eval(model.prompt_text(prompt_str))
    n_act = len(act_list)

    segments = []
    aug_cap_set = set()
    for k in range(args.n_sample):
        cap = seg["caption"]
        np.random.shuffle(act_list)
        n_act_replace = np.random.choice(n_act + 1)
        wrong_acts = []
        originals = []
        augments = []
        for i in range(n_act_replace):
            act = act_list[i]
            prompt_str = PROMPT_SET["new_act"].format(act=act)
            act_new = model.prompt_text(prompt_str)
            wrong_acts.append(act_new)
            prompt_str = PROMPT_SET["substitute"].format(ow=act, nw=act_new, cap=cap)
            cap = model.prompt_text(prompt_str)
            originals.append(act.strip())
            augments.append(act_new.strip())

        if cap in aug_cap_set:
            continue
        aug_cap_set.add(cap)
        quality = round(1 - float(n_act_replace) / (n_act + 1), 4)
        scores, similarity, quality = get_score(
            originals, augments, seg["caption"], cap
        )
        explanation = {"wrong actions": wrong_acts}
        segments.append(
            {
                "timestamp": [st, end],
                "caption": cap,
                "quality": quality,
                "explanation": explanation,
                "original_objects": originals,
                "augmented_objects": augments,
                "original_caption": seg["caption"],
                "augmented_caption": cap,
                "obj_vs_obj": scores,
                "caption_vs_caption": similarity,
            }
        )
    return segments


def change_obj_act(args, seg, model):
    st, end = seg["timestamp"]
    cap = seg["caption"]
    # parse objects in the caption
    prompt_str = PROMPT_SET["obj_list"].format(cap=cap)
    obj_list = eval(model.prompt_text(prompt_str))
    n_obj = len(obj_list)

    # parse actions in the caption
    prompt_str = PROMPT_SET["act_list"].format(cap=cap)
    act_list = eval(model.prompt_text(prompt_str))
    n_act = len(act_list)

    segments = []
    aug_cap_set = set()
    for k in range(args.n_sample):
        cap = seg["caption"]
        np.random.shuffle(obj_list)
        n_obj_replace = np.random.choice(n_obj + 1)
        wrong_objs = []
        originals = []
        augments = []
        for i in range(n_obj_replace):
            obj = obj_list[i]
            prompt_str = PROMPT_SET["new_obj"].format(obj=obj)
            obj_new = model.prompt_text(prompt_str)
            wrong_objs.append(obj_new)
            prompt_str = PROMPT_SET["substitute"].format(ow=obj, nw=obj_new, cap=cap)
            cap = model.prompt_text(prompt_str)
            originals.append(obj.strip())
            augments.append(obj_new.strip())

        np.random.shuffle(act_list)
        n_act_replace = np.random.choice(n_act + 1)
        wrong_acts = []
        for i in range(n_act_replace):
            act = act_list[i]
            prompt_str = PROMPT_SET["new_act"].format(act=act)
            act_new = model.prompt_text(prompt_str)
            wrong_acts.append(act_new)
            prompt_str = PROMPT_SET["substitute"].format(ow=act, nw=act_new, cap=cap)
            cap = model.prompt_text(prompt_str)
            originals.append(act.strip())
            augments.append(act_new.strip())

        if cap in aug_cap_set:
            continue
        aug_cap_set.add(cap)
        quality = round(1 - float(n_obj_replace + n_act_replace) / (n_obj + n_act), 4)
        scores, similarity, quality = get_score(
            originals, augments, seg["caption"], cap
        )
        explanation = {"wrong objects": wrong_objs, "wrong actions": wrong_acts}
        segments.append(
            {
                "timestamp": [st, end],
                "caption": cap,
                "quality": quality,
                "explanation": explanation,
                "original_objects": originals,
                "augmented_objects": augments,
                "original_caption": seg["caption"],
                "augmented_caption": cap,
                "obj_vs_obj": scores,
                "caption_vs_caption": similarity,
            }
        )
    return segments


def augment(args, ann):
    global sen_model
    sen_model = OpenAI(
        base_url="http://localhost:31090/v1",
        api_key="EMPTY",
    )
    model = VLLM(args)
    vid = ann["vid"]
    segments = []
    for seg in ann["segments"]:
        aug_seg = {}
        aug_seg["gt"] = [seg]
        try:
            funcs = [change_obj, change_act, change_obj_act]
            results = Parallel(n_jobs=args.num_of_workers, prefer="threads")(
                delayed(func)(args, seg, model) for func in funcs
            )
            aug_seg["change_obj"] = results[0]
            aug_seg["change_act"] = results[1]
            aug_seg["change_obj_act"] = results[2]

            segments.append(aug_seg)
        except Exception as e:
            logger.error(f"Error in augmenting segment {seg}: {e}")
            pass

    return vid, segments
