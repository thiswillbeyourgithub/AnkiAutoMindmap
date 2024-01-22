import litellm
import numpy as np
from joblib import Memory
from tqdm import tqdm

from utils.logger import red

litellm.set_verbose = False

def embedder(text_list, modelname, result):
    """compute the emebdding of 1 text
    if result is not None, it is the embedding and returned right away. This
    was done to allow caching individual embeddings while still making one batch
    call to the embedder.
    """
    assert isinstance(text_list, list)
    if result is not None:
        assert isinstance(result, list)
        assert len(text_list) == len(result)
        assert all(isinstance(a, np.ndarray) for a in result)
        result = [r.reshape(1, -1) for r in result]
        return result

    vec = litellm.embedding(
            model=modelname,
            input=text_list,
            )
    return [np.array(d["embedding"]).reshape(1, -1) for d in vec.data]


def embedder_wrapper(list_text, embedmodel):
    # both the embedder and the wrapper are cached
    mem = Memory(f"cache/{embedmodel}", verbose=0)
    cached_embedder = mem.cache(embedder, ignore=["result"])
    uncached_texts = [t
                      for t in list_text
                      if not cached_embedder.check_call_in_cache([t], embedmodel, None)]

    if not uncached_texts:
        red("Everything already in cache")
        return [cached_embedder([t], embedmodel, None) for t in tqdm(list_text, desc="loading from cache")]

    if len(uncached_texts) > 1:
        present = len(list_text) - len(uncached_texts)
        red(f"Embeddings present in cache: {present}/{len(list_text)}")

    results = cached_embedder(uncached_texts, embedmodel, None)

    # manually recache the values for each individual memory
    [cached_embedder([t], embedmodel, [r]) for t, r in zip(tqdm(uncached_texts, desc="Setting cache"), results)]

    # combine cached and uncached results in the right order
    to_return = []
    it_results = iter(results)
    cnt = 0
    for i in range(len(list_text)):
        if list_text[i] in uncached_texts:
            to_return.append(next(it_results)[0])
            cnt += 1
        else:
            to_return.append(cached_embedder([list_text[i]], embedmodel, None)[0])
    # make sure the list was emptied
    assert cnt == len(results)
    return to_return


def llmcall(messages, llmname):
    """calls LLM"""
    out = litellm.completion(
            model=llmname,
            temperature=0,
            messages=messages,
            ).json()
    return out
