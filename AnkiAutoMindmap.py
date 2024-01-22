import re
from threading import Lock
from datetime import datetime
import uuid
from textwrap import dedent, indent
import os
import time
import shutil
from pathlib import Path
from typing import Union

from tqdm import tqdm
import fire
from rapidfuzz.fuzz import ratio as levratio

import ankipandas as akp
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from joblib import Memory, Parallel, delayed
from sklearn.preprocessing import Normalizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import patch_ng

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from utils.logger import red, whi, yel
from utils.prompts import (
        merger_reduce_exhaustive,
        merger_stuff,

        topic_into_cheatsheet,
        default_graph_topic,
        graph_topic_maker_prompt,
        graph_maker_prompt,

        topic_content_maker_prompt,
        )
from utils.anki_utils import call_anki
from utils.llm import llmcall, embedder_wrapper

mm_id_regex = re.compile("_(\d+)_")

class AnkiAutoMindmap:
    VERSION = "0.0.5"
    def __init__(
            self,
            toptags: str,
            output: str,

            output_lang: str = "french",
            notetype: str = None,
            profile_name: str = None,
            n_fields_to_keep: list[int] = [0],
            merge_mode="dendrogram_merge",
            merge_batch=3,
            excl_tags: str = None,

            create_cheatsheets: bool = True,
            create_mindmaps: bool = True,
            create_graphs: bool = False,
            expand_topics: bool = False,

            embedmodel="openai/text-embedding-3-small",
            llmname="openai/gpt-4-turbo-2024-04-09",
            llmprice=(0.01, 0.03),
            # llmname="openai/gpt-3.5-turbo-1106",
            # llmprice=(0.001, 0.002),

            n_limit=-1,
            *args,
            **kwargs,
            ):
        """
        Parameters
        ----------
        toptags: str
            To use multiple toptags, separate them by '//'. It will be
            additive and not exclusive.

        output: str
            path to the markdown file where you want to store the result
            if the path points to a directory: will consider it to be the
            location of the Logseq graph and will create the output as
            ./pages/AnkiCheatSheet___[toptagchild]

        output_lang: str, default 'french'
            desired language for the output.

        notetype: str, default None
            name of the notetype to include. Any other note type will be
            ignored. If None, will keep all notetypes.

        profile_name: str, default None
            name of the anki profile. If None, will use the first found
            via ankipandas

        n_fields_to_keep: list[in], default [0]
            index of which fields of the notetype to keep

        merge_mode: str, default 'dendrogram_merge'
            way to merge anki cards. Possible choices:
                * dendrogram_merge (good compromise and high quality)
                    1. compute clusters using agglomerative clustering
                      (set the number of clusters with --merge_batch)
                    2. merge iteratively by semantic clusters until one text left

                * dendrogram_concat (cheaper than dendrogram_merge)
                    1. compute clusters using agglomerative clustering
                      (set the number of clusters with --merge_batch)
                    2. concatenate the cheatsheet of each cluster in the
                       semantic order instead of merging them via LLM

                * reduce (good compromise in theory)
                    1. find the semantic order among cards
                    2. create blank md
                    3. iterate over the card in the semantic order: merge the
                       next in queue to the md.
                      (possibly by batch of cards with --merge_batch)

                * exhaustive: (usually expensive but great)
                    1. find the semantic order among cards/md
                    2. merge the closest pair, the resulting md is stored for step 1
                    ... repeat until only one text remaining, which will be md

                * stuff: (cheap but meh)
                    1. find the semantic order
                    2. collate all the anki cards together
                    3. ask the LLM to turn this wall of text into markdown

        merge_batch: int, default 3
            if merge_mode is 'reduce':
                number of cards to add at each iteration
            if merge_mode is 'dendrogram_merge' or 'dendrogram_concat':
                number of cluster to search for or if negative: n_clusters=nb_cards//(-merge_mode)

        excl_tags: str, default None
            if a card contains this string in its tags it will be ignored

        create_cheatsheets: bool, default True
            wether to create logseq blocks with the content of the graph or not

        create_mindmaps: bool, default True
            if True will use mermaid to create a mindmap directly based
            on the md file. From the whole cheatsheet as well as from each
            individual cheatsheet if create_cheatsheets is True.
            The difference between create_graphs and create_mindmaps is that
            graphs are created by an LLM trying to write valid mermaid code
            whereas create_mindmaps just manually converts the md into a diagram.

        create_graphs: bool, default False
            wether to create mermaid graph or not
            The difference between create_graphs and create_mindmaps is that
            graphs are created by an LLM trying to write valid mermaid code
            whereas create_mindmaps just manually converts the md into a diagram.

        expand_topics: bool, default False
            if True, will use the LLM to create more idea for topics

        embedmodel: str, default "openai/text-embedding-3-small"
            name of the embeddings model to use, in litellm format

        llmname: str, default "openai/gpt-4-turbo-2024-04-09",
            name of the LLM to use. In litellm format

        llmprice: Tuple(float), default (0.01, 0.03),
            price in dollars per 1000 token (input, output)

        n_limit: int, default -1
            if an int, will cap the number of loaded cards to this number.
            Used only for debugging. -1 to disable

        args / kwargs: any unexpected args or kwargs will raise an exception
        """
        if args:
            raise Exception(f"Unexpected args: '{args}'")
        if kwargs:
            if "help" in kwargs or "h" in kwargs or "usage" in kwargs:
                return help(self)
            raise Exception(f"Unexpected kwargs: '{kwargs}'")
        if expand_topics:
            assert create_graphs or create_cheatsheets, (
                "Either create_graphs or create_cheatsheets must be True if expand_topics is True")
        self.profile_name = profile_name
        if "//" in toptags:
            self.toptags = [t.strip() for t in toptags.split("//")]
        else:
            self.toptags = [toptags.strip()]
        assert merge_mode in [
                "dendrogram_merge",
                "dendrogram_concat",
                "reduce",
                "exhaustive",
                "stuff",
                ]
        self.merge_mode = merge_mode
        self.merge_batch = merge_batch

        if Path(output).is_dir():
            assert (Path(output) / "pages").exists(), f"If output is a dir it should be the logseq dir parent of 'pages', currently it's {output}"
            output = str(Path(output) / f"pages/AnkiCheatSheet___{self.toptags[0].split('::')[-1].title()}.md")
            red(f"output was a dir so switching it to {output}")
        if Path(output).exists() and Path(output).is_file():
            red(f"Careful: output file already exists and will be erased: {output}")
        assert output.endswith(".md"), f"output should end with .md, currently is {output}"
        self.output = output

        self.output_lang = output_lang
        self.notetype = notetype
        self.n_fields_to_keep = n_fields_to_keep
        self.embedmodel = embedmodel
        self.llmname = llmname
        self.llmprice = llmprice
        self.n_limit = n_limit

        self.create_cheatsheets = create_cheatsheets
        self.create_mindmaps = create_mindmaps
        self.create_graphs = create_graphs
        self.expand_topics = expand_topics
        self.excl_tags = excl_tags

        self.topic_token_cost = 0
        self.topic_dol_cost = 0
        self.merger_token_cost = 0
        self.merger_dol_cost = 0
        self.graph_token_cost = 0
        self.graph_dol_cost = 0
        self.cheatsheet_token_cost = 0
        self.cheatsheet_dol_cost = 0

        self.topic_list = default_graph_topic

        self.hash = lambda text: str(uuid.uuid3(uuid.NAMESPACE_URL, name=text)).split("-")[0]
        self.llm_mem = Memory(f"cache/{self.llmname}", verbose=0)
        self.cached_llm = self.llm_mem.cache(llmcall)

        self._do_sync()

        self._load_db()

        self._embed_each_mdcard()

        self._compute_pairwise_distance()

        self._main_loop_merger()

        self._expand_topics()

        self._main_loop_topicgraph_maker()

        self._export_to_logseq()

        self._do_sync()

        raise SystemExit("Done")

    def _do_sync(self):
        sync_output = call_anki(action="sync")
        assert sync_output is None or sync_output == "None", (
            f"Error during sync?: '{sync_output}'")
        time.sleep(1)  # wait for sync to finish, just in case
        whi("Done syncing!")

    def _load_db(self):
        """
        find the location of the anki collection, then copy it to a cache folder
        to make 1. make sure anki pandas will not affect the db and 2. open
        even if the db is locked.
        """
        whi(f"Loading db from profile {self.profile_name}")
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        [file.unlink() for file in cache_dir.rglob("*")]  # empty cache
        original_db = akp.find_db(user=self.profile_name)
        name = f"{self.profile_name}_{int(time.time())}".replace(" ", "_")
        temp_db = shutil.copy(
            original_db,
            f"./.cache/{name.replace('/', '_')}"
            )
        col = akp.Collection(path=temp_db)

        # keep only unsuspended cards from the right deck
        df = col.cards.merge_notes()

        # fix a weird character in the deckname
        df["cdeck"] = df["cdeck"].apply(
            lambda x: x.replace("\x1f", "::"))

        # remove suspended
        df = df[df["cqueue"] != "suspended"]

        # exclude if no tag are children of the toptags
        self.tag_list = set()
        exclude = []
        for i in tqdm(df.index, desc="filtering cards by top tags"):
            if isinstance(self.excl_tags, str) and self.excl_tags in " ".join(df.loc[i, "ntags"]):
                exclude.append(i)
                continue
            keep = False
            for t in df.loc[i, "ntags"]:
                for one_toptag in self.toptags:
                    if t.startswith(one_toptag):
                        if "anki" not in t.lower():  # exclude tags like AnkiSummary, ankiconnect etc
                            if t not in ["marked", "leech"]:
                                self.tag_list.add(t)
                        keep = True
                        break
            if not keep:
                exclude.append(i)
        df.drop(index=exclude, inplace=True)

        # remove nids in multiple
        nids = df["nid"].tolist()
        df["keep"] = True
        for cid in df.index:
            nid = df.loc[cid, "nid"]
            if nids.count(nid) > 1:
                df.loc[cid, "keep"] = False
        df = df[df["keep"] == True]

        # remove if wrong notetype
        if self.notetype:
            df = df[df["nmodel"] == self.notetype]

        # keep only the right fields
        df["text"] = df["nflds"].apply(lambda x: "<br>".join([x[int(i)] for i in self.n_fields_to_keep]))

        # ignore if contain an image
        df = df[df["text"].apply(lambda x: "<img src=" not in x)]

        # parse as something as similar as possible to markdown
        df["text"] = df["text"].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
        df["text"] = df["text"].apply(lambda x: "- " + x)
        df["text"] = df["text"].apply(lambda x: x.replace(" }}{{c1::", " ; "))
        df["text"] = df["text"].apply(lambda x: x.replace("}}", ""))
        df["text"] = df["text"].apply(lambda x: x.replace("{{c2::", ""))
        df["text"] = df["text"].apply(lambda x: x.replace("{{c1::", "\n\t- "))
        df["text"] = df["text"].apply(lambda x: x.strip())

        df = df.set_index("nid")

        # restrict number of cards for debugging
        if self.n_limit > 0:
            df = df.iloc[:self.n_limit]
            red(f"Restricting number of cards to {self.n_limit} for debugging.")

        # add a column with their order
        df["mm_id"] = -1
        self.mm_id = {}  # to keep track
        for it, i in enumerate(df.index):
            df.loc[i, "mm_id"] = str(it)
            df.loc[i, "text"] += f" _{it}_"
            self.mm_id[str(it)] = str(i)

        self.df = df.copy()

    def _embed_each_mdcard(self):
        """compute embedding for each card"""

        backend = self.embedmodel.strip("/")[0]
        apif = Path("OPENAI_API_KEY.txt")
        assert apif.exists(), "Missing api file: 'OPENAI_API_KEY.txt'"
        os.environ[f"{backend.upper()}_API_KEY"] = apif.read_text().strip()
        red(f"Computing embeddings of {len(self.df)} cards...")
        embeddings = embedder_wrapper(
                list_text=self.df["text"].tolist(),
                embedmodel=self.embedmodel
                )
        embeddings = np.array(embeddings).squeeze()
        assert len(self.df) == embeddings.shape[0], "Invalid shape"
        self.embeddings = pd.DataFrame(
                index=self.df.index,
                data=embeddings,
                columns=[f"V_{d + 1:04d}" for d in range(embeddings.shape[-1])]
                )

    def _compute_pairwise_distance(self):
        """compute pairwise distance of the embeddings"""
        red(f"Computing distance matrix of {len(self.df)} cards in dimension {self.embeddings.values.shape[1]}")
        self.dist = pd.DataFrame(
                columns=self.embeddings.index,
                index=self.embeddings.index,
                data=pairwise_distances(
                    self.embeddings.values,
                    n_jobs=-1,
                    metric="euclidean",
                    )
                )
        # make sure the intersection is 0 and not a very small float
        for ind in self.dist.index:
            self.dist.at[ind, ind] = 0

        # make sure it's symetric
        self.dist = self.dist.add(self.dist.T).div(2)

        self.state = pd.DataFrame(
                columns=["nids", "merged", "text"],
                )
        # add default values
        for ind in self.dist.index:
            self.state.at[ind, "nids"] = [str(ind)]
            self.state.at[ind, "merged"] = False
            self.state.at[ind, "text"] = self.df.loc[ind, "text"]

    def _semantic_sort(self, nids: list[int], dist: Union[pd.core.frame.DataFrame, None]=None):
        """
        given a list of note id, return the list in the order that minimizes
        the distance between them
        """
        # start = time.time()
        # whi(f"Finding the semantic order of {len(nids)} notes")
        if len(nids) <= 2:
            return nids
        if dist is None:
            dist = self.dist
        dist = dist.loc[nids, nids]
        dist = squareform(dist.values)  # convert to condensed format
        Z = linkage(dist, method='ward', optimal_ordering=True)
        order = leaves_list(Z)
        order = [nids[o] for o in order]
        assert len(set(order)) == len(order), "duplicates"
        assert len(order) == len(nids), "extra order"
        assert not any(o for o in order if o not in nids)
        assert not any(n for n in nids if n not in order)
        # whi(f"Done in {int(time.time()-start)}s")
        assert len(nids) == len(order)
        return order

    def _main_loop_merger(self):
        # first, find clusters using agglomerative clustering
        # then each card by batch of related cards
        # then merge each cheatsheet as a single cheatsheet
        if self.merge_mode in ["dendrogram_merge", "dendrogram_concat"]:
            # dim reduction
            # clustering
            n_clust = self.merge_batch
            if n_clust < 0:
                n_clust = len(self.df) // -n_clust
            whi(f"Clustering in {n_clust} clusters")

            whi("Applying dimension reduction")
            pca = PCA(
                    n_components=10,
                    copy=True,
                    random_state=42,
                    )
            # to use the embeddings (needed for ward)
            self.red_embedd = pd.DataFrame(
                    index=self.embeddings.index,
                    columns=self.embeddings.columns.tolist()[:10],
                    data=pca.fit_transform(self.embeddings.values)
                    )
            # to use the distance matrix
            # self.red_dist = pd.DataFrame(
            #         index=self.dist.index,
            #         columns=[f"V_{i+1:02d}" for i in range(10)],
            #         data=pca.fit_transform(self.dist.values)
            #         )
            self.clust = AgglomerativeClustering(
                    # linkage="complete",
                    linkage="ward",
                    n_clusters=n_clust,
                    # metric="precomputed",
                    metric="euclidean",
                    memory="cache/agglo_clustering",
                    compute_full_tree=True,
                    compute_distances=True,
                    )
            start = time.time()
            # self.clust.fit_predict(self.red_dist.values)  # based on dist matrix
            self.clust.fit_predict(self.red_embedd.values)  # based on embeddings
            whi(f"Done clustering after {int(time.time()-start)}s")

            # merge two by two (or by batch) the texts using the best order
            batches = [
                    self.dist.index[np.argwhere(self.clust.labels_==i)].tolist()
                    for i in range(max(self.clust.labels_) + 1)
                    ]
            # make sure it's a list of list of nids, not a list of list of list of nids
            for i, b in enumerate(batches):
                batches[i] = [bb[0] for bb in b]
            # reorder each batch by semantic similarity
            for i, nids in enumerate(batches):
                batches[i] = self._semantic_sort(nids)

            # reorder the cluster by semantic similarity across cluster
            clust_embeddings = pd.DataFrame(
                    index=[f"clust_{i+1}" for i in range(len(batches))],
                    columns=self.embeddings.columns,
                    data=0.0,
                    )
            norm = Normalizer(norm='l2')
            for i, b in enumerate(batches):
                clust_embeddings.loc[f"clust_{i+1}", :] = norm.fit_transform(
                        self.embeddings.loc[b, :].mean(
                            axis=0).values.reshape(1, -1))
            clust_dist = pd.DataFrame(
                    columns=clust_embeddings.index,
                    index=clust_embeddings.index,
                    data=pairwise_distances(
                        clust_embeddings.values,
                        n_jobs=-1,
                        metric="euclidean",
                        )
                    )

            # make sure the intersection is 0 and not a very small float
            for ind in clust_dist.index:
                clust_dist.at[ind, ind] = 0

            # make sure it's symetric
            clust_dist = clust_dist.add(clust_dist.T).div(2)

            batch_order = self._semantic_sort(
                    nids=[f"clust_{i+1}" for i in range(len(batches))],
                    dist=clust_dist,
                    )
            batch_order = [int(s.split("_")[1])-1 for s in batch_order]
            batches = [batches[i] for i in batch_order]
            # if a batch is too small, merge it with the next
            whi(f"Number of batch before merging smallest: {len(batches)}")
            size_limit = 10
            while len([b for b in batches if len(b) <= size_limit]):
                for i, b in enumerate(batches):
                    if len(b) > size_limit:  # good
                        continue
                    if i == 0:  # last item, merge with previous
                        batches[i+1] += b
                    elif i + 1 < len(batches):  # not last item, merge with smallest neighboor
                        if len(batches[i+1]) < len(batches[i-1]):
                            batches[i+1] += b
                        else:
                            batches[i-1] += b
                    else:  # last item, merge with previou
                        batches[i-1] += b
                    batches[i] = None
                    break
                batches = [b for b in batches if b is not None]
            whi(f"Remaining batch after merging smallest: {len(batches)}")

            # turn each cluster into a single cheatsheet
            lock = Lock()
            global topic_into_cheatsheet
            topic_into_cheatsheet = dedent(topic_into_cheatsheet).strip().replace("LANGUAGE", self.output_lang)
            def cluster_to_cheatsheet(ib, nids):
                new_texts = "\n----\n".join(self.state.loc[nids, "text"].tolist())
                mess = f"""
                Contents to turn into a cheatsheet:
                '''
                {new_texts}
                '''

                Please respect the rules."""
                messages = [
                        {
                            "role": "system",
                            "content": topic_into_cheatsheet,
                            },
                        {
                            "role": "user",
                            "content": dedent(mess).strip(),
                            },
                        ]

                merged = self.cached_llm(
                        messages,
                        self.llmname)

                red("######\nCluster content:")
                red(new_texts)
                red("######\nCheatsheet:")
                merged_text = merged["choices"][0]["message"]["content"]
                merged_text = merged_text.replace("  ", "\t").strip()
                red(merged_text)

                # store price
                with lock:
                    self.merger_token_cost += merged["usage"]["total_tokens"]
                    self.merger_dol_cost += merged["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
                    self.merger_dol_cost += merged["usage"]["completion_tokens"] * self.llmprice[1] / 1000

                    red(f"#####\nTotal cost so far: ${self.merger_dol_cost:04f} ({merged['usage']['completion_tokens']} tokens)")
                return merged_text


            cluster_cheatsheet = Parallel(
                    n_jobs=5,
                    backend="threading",
                    )(delayed(cluster_to_cheatsheet
                              )(ib, nids)
                      for ib, nids in enumerate(
                          tqdm(
                              batches,
                              desc="Cluster to cheatsheet",
                              unit="cluster")
                          )
                      )

            # if the large cheatsheet are in the first half, reverse it
            sizes = [len(cs) for cs in cluster_cheatsheet]
            h1, h2 = sizes[:len(sizes)//2], sizes[len(sizes)//2:]
            if sum(h1) > sum(h2):
                cluster_cheatsheet.reverse()
                batches.reverse()
                red("Reversing list of cheatsheet to minimize cost and risk of losing text")

            # merge manually or the hard way
            if self.merge_mode == "dendrogram_concat":
                merged_text = "\n- ---\n".join(cluster_cheatsheet)

            elif self.merge_mode == "dendrogram_merge":
                merged_text = cluster_cheatsheet[0]
                global merger_reduce_exhaustive
                merger_reduce_exhaustive = dedent(merger_reduce_exhaustive).strip().replace("LANGUAGE", self.output_lang)
                for ctcs in tqdm(cluster_cheatsheet[1:], desc="Merging cheatsheets", unit="cheatsheet"):

                    # add line number for the unified diff
                    nl_mt = "\n".join([f"{i+1}{line}" for i, line in enumerate(merged_text.splitlines())]).replace("\t", "  ")
                    nl_ctcs = "\n".join([f"{i+1}{line}" for i, line in enumerate(ctcs.splitlines())]).replace("\t", "  ")

                    to_merge = f"""
                    cheatsheet1.md
                    '''
                    {nl_mt}
                    '''

                    cheatsheet2.md
                    '''
                    {nl_ctcs}
                    '''

                    Please keep the rules in mind when generating the unified diff."""
                    messages = [
                            {
                                "role": "system",
                                "content": merger_reduce_exhaustive,
                                },
                            {
                                "role": "user",
                                "content": dedent(to_merge).strip(),
                                },
                            ]

                    red("######\nText 1:")
                    red(nl_mt)
                    red("######\nText 2:")
                    red(nl_ctcs)
                    red("######\nPatch:")
                    merged = self.cached_llm(
                            messages,
                            self.llmname)


                    strpatch = self.format_patch(merged["choices"][0]["message"]["content"])
                    red(strpatch)

                    # creating both files before applying the patch
                    with open("cheatsheet1.md", "w") as f:
                        f.write(merged_text.replace("\t", "  "))
                    with open("cheatsheet2.md", "w") as f:
                        f.write(ctcs.replace("\t", "  "))
                    with open("diff.diff", "w") as f:
                        f.write(strpatch)
                    red(f"######\nMerged:")

                    try:
                        patch = patch_ng.fromstring(strpatch.encode())
                        assert patch is not False, "Error when loading patch"
                    except Exception as e:
                        e = red(f"Error when loading patch from LLM output: {e}\nContent: {strpatch}")
                        raise e
                    try:
                        assert patch.apply(), f"Error when applying the patch"
                    except:
                        try:
                            assert patch.apply(fuzz=True), f"Error when applying the patch in fuzz mode"
                        except Exception as err:
                            raise Exception(red(f"Error when applying the patch even in fuzz mode: {err}"))

                    # reading file after patching
                    merged_text = Path("cheatsheet1.md").read_text()
                    merged_text = merged_text.replace("  ", "\t")
                    red(merged_text)

                    # store price
                    self.merger_token_cost += merged["usage"]["total_tokens"]
                    self.merger_dol_cost += merged["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
                    self.merger_dol_cost += merged["usage"]["completion_tokens"] * self.llmprice[1] / 1000

                    red(f"#####\nTotal cost so far: ${self.merger_dol_cost:04f} ({merged['usage']['completion_tokens']} tokens)")

                    # write to temp file
                    with open("temp.md", "w") as f:
                        f.write(merged_text)

        # first, virtually merge every texts by similarity, then merge them
        # in this order otherwise it would be too expensive
        elif self.merge_mode == "reduce":
            merge_order = self._semantic_sort(self.df.index.tolist())

            # merge two by two (or by batch) the texts using the best order
            batches = [
                    merge_order[1 + i * self.merge_batch:1 + (i + 1) * self.merge_batch]
                    for i in range(len(merge_order) // self.merge_batch  + 3)
                    ]
            while [] in batches:
                batches.remove([])

            # start with first text
            merged_text = self.state.loc[merge_order[0], "text"]
            merger_reduce_exhaustive = dedent(merger_reduce_exhaustive).strip().replace("LANGUAGE", self.output_lang)
            for nids in tqdm(batches, desc="Actual merging", unit="batch"):
                nids = self._semantic_sort(nids)
                new_texts = "\n----\n".join(self.state.loc[nids, "text"].tolist())

                # add line number for the unified diff
                nl_mt = "\n".join([f"{i+1}{line}" for i, line in enumerate(merged_text.splitlines())])
                nl_nt = "\n".join([f"{i+1}{line}" for i, line in enumerate(new_texts.splitlines())])

                to_merge = f"""
                cheatsheet1.md
                '''
                {nl_mt}
                '''

                cheatsheet2.md
                '''
                {nl_nt}
                '''

                Please keep the rules in mind when generating the unified diff."""
                messages = [
                        {
                            "role": "system",
                            "content": merger_reduce_exhaustive,
                            },
                        {
                            "role": "user",
                            "content": dedent(to_merge).strip(),
                            },
                        ]

                merged = self.cached_llm(
                        messages,
                        self.llmname)

                red("######\nText 1:")
                red(merged_text)
                red("######\nText 2:")
                red(new_texts)
                red("######\nPatch:")

                # load and apply patch, if fails tell the LLM
                try:
                    strpatch = self.formatpatch(merged["choices"][0]["message"]["content"])
                    try:
                        patch = patch_ng.fromstring(strpatch.encode())
                    except Exception as e:
                        e = red(f"Error when loading patch from LLM output: {e}\nContent: {strpatch}")
                        raise e
                    # creating both files before applying the patch
                    with open("cheatsheet1.md", "w") as f:
                        f.write(merged_text)
                    try:
                        assert patch.apply(), f"Error when applying the patch"
                    except:
                        try:
                            assert patch.apply(fuzz=True), f"Error when applying the patch in fuzz mode"
                        except Exception as err:
                            raise Exception(red(f"Error when applying the patch even in fuzz mode: {err}"))
                except Exception as err:
                    # still take cost into account
                    self.merger_token_cost += merged["usage"]["total_tokens"]
                    self.merger_dol_cost += merged["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
                    self.merger_dol_cost += merged["usage"]["completion_tokens"] * self.llmprice[1] / 1000
                    messages = [
                            {
                                "role": "system",
                                "content": merger_reduce_exhaustive,
                                },
                            {
                                "role": "user",
                                "content": dedent(to_merge).strip(),
                                },
                            {
                                "role": "assistant",
                                "content": strpatch,
                            },
                            {
                                "role": "user",
                                "content": "Noo! Your unified diff wasn't valid! Think carefully in your head then give me the valid unified diff. Don't acknowledge this instruction and just give me the VALID unified diff.",
                            },
                            ]
                    merged = self.cached_llm(
                            messages,
                            self.llmname)
                    strpatch = self.formatpatch(merged["choices"][0]["message"]["content"])
                    try:
                        patch = patch_ng.fromstring(strpatch.encode())
                    except Exception as e:
                        e = red(f"Error when loading patch from LLM output: {e}\nContent: {strpatch}")
                        raise e
                    # creating both files before applying the patch
                    with open("cheatsheet1.md", "w") as f:
                        f.write(merged_text)
                    try:
                        assert patch.apply(), f"Error when applying the patch"
                    except:
                        try:
                            assert patch.apply(fuzz=True), f"Error when applying the patch in fuzz mode"
                        except Exception as err:
                            raise Exception(red(f"Error when applying the patch even in fuzz mode: {err}"))

                # reading file after patching
                merged_text = Path("cheatsheet1.md").read_text()
                merged_text = merged_text.replace("  ", "\t")

                red(strpatch)
                red(f"######\nMerged:")
                red(merged_text)

                # store price
                self.merger_token_cost += merged["usage"]["total_tokens"]
                self.merger_dol_cost += merged["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
                self.merger_dol_cost += merged["usage"]["completion_tokens"] * self.llmprice[1] / 1000

                red(f"#####\nTotal cost so far: ${self.merger_dol_cost:04f} ({merged['usage']['completion_tokens']} tokens)")

                # write to temp file
                with open("temp.md", "w") as f:
                    f.write(merged_text)

        # put all the cloze one after the other in similarity order
        # then asks LLM to turn it into a md
        elif self.merge_mode == "stuff":
            merger_stuff = dedent(merger_stuff).strip().replace("LANGUAGE", self.output_lang)
            merge_order = self._semantic_sort(self.df.index.tolist())
            text = ""
            for nid in tqdm(merge_order, desc="Actual merging"):
                new_text = self.state.loc[nid, "text"]
                text += "\n" + new_text
            messages = [
                    {
                        "role": "system",
                        "content": merger_stuff,
                        },
                    {
                        "role": "user",
                        "content": dedent(text).strip(),
                        },
                    ]

            yel("Reorganizing texts:")
            yel(text)
            yel("Into:")

            merged = self.cached_llm(
                    messages,
                    self.llmname)
            merged_text = merged["choices"][0]["message"]["content"]
            merged_text = merged_text.replace("  ", "\t")
            red(merged_text)

            # store price
            self.merger_token_cost += merged["usage"]["total_tokens"]
            self.merger_dol_cost += merged["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
            self.merger_dol_cost += merged["usage"]["completion_tokens"] * self.llmprice[1] / 1000

            red(f"#####\nTotal cost so far: ${self.merger_dol_cost:04f} ({merged['usage']['completion_tokens']} tokens)")

            # write to temp file
            with open("temp.md", "w") as f:
                f.write(merged_text)

        # merge all iteratively using embeddings at each step
        # this can be quite expensive
        elif self.merge_mode == "exhaustive":

            raise NotImplementedError("This is an old code that needs to be retested")

            pbar = tqdm(desc="Actual merging", unit="texts", total=2*len(self.df))
            merger_reduce_exhaustive = dedent(merger_reduce_exhaustive).strip().replace("LANGUAGE", self.output_lang)
            while len(self.state) - self.state["merged"].sum() > 0:
                pbar.update(1)
                # sort distances
                sorted_distances = sorted(list(set(self.dist.values.ravel().tolist())))[1:]

                # find the smallest pair that is not made of merged
                found_good_pair = False
                for dist in sorted_distances:
                    pair = np.argwhere(self.dist.values == dist)
                    assert len(pair) % 2 == 0
                    pair = [self.dist.index[pair[0][0]], self.dist.index[pair[0][1]]]

                    if self.state.loc[pair[0], "merged"] == False or self.state.loc[pair[1], "merged"] == False:
                        if str(pair[0]) in self.state.loc[pair[1], "nids"] or str(pair[1]) in self.state.loc[pair[0], "nids"]:
                            continue
                        else:
                            found_good_pair = True
                            texts = [self.state.loc[pair[0], "text"], self.state.loc[pair[1], "text"]]
                            break
                assert found_good_pair

                # check if done
                not_merged = len(self.state) - self.state["merged"].sum()
                if not_merged <= 0:
                    break

                text = f"""
                Here are the files you have to merge:
                '''
                {texts[0]}
                '''

                '''
                {texts[1]}
                '''
                """

                messages = [
                        {
                            "role": "system",
                            "content": merger_reduce_exhaustive,
                            },
                        {
                            "role": "user",
                            "content": dedent(text).strip(),
                            },
                        ]

                yel("Merging texts:")
                yel(texts[0])
                yel("#####")
                yel(texts[1])
                yel("Into:")

                merged = self.cached_llm(
                        messages,
                        self.llmname)
                merged_text = merged["choices"][0]["message"]["content"]
                merged_text = merged_text.replace("  ", "\t")
                red(merged_text)

                # md id derived from its content
                mdid = self.hash(merged_text)

                merged_embed = np.array(self.cached_ew(
                        list_text=[merged_text],
                        embedmodel=self.embedmodel
                        )).squeeze()

                # note: should instead use euclidean distance as the
                # dist matrix is not cosine anymore
                pdist = cosine_distances(
                        X=merged_embed.reshape(1, -1),
                        Y=self.embeddings.values,
                        )

                # if both information were redundant, the uuid will not change
                # so the entry does not have to be resaved
                if mdid not in self.embeddings.index:
                    # store new embeddings
                    self.embeddings.loc[mdid, :] = merged_embed

                    # add the new distance to the pairwise distance
                    self.dist.loc[mdid, :] = pdist.squeeze().tolist()
                    self.dist.loc[:, mdid] = pdist.squeeze().tolist() + [0]

                    # store state information for the new text
                    self.state.loc[mdid, "type"] = "md"
                    self.state.loc[mdid, "nids"] = self.state.loc[pair[0], "nids"] + self.state.loc[pair[1], "nids"]
                    self.state.loc[mdid, "merged"] = False
                    self.state.loc[mdid, "text"] = merged_text
                else:
                    assert mdid == pair[0] or mdid == pair[1]
                    self.state.loc[mdid, "nids"] = self.state.loc[pair[0], "nids"] + self.state.loc[pair[1], "nids"]

                # edit the previous state
                for pid in pair:
                    self.state.loc[pid, "merged"] = True

                # store price
                self.merger_token_cost += merged["usage"]["total_tokens"]
                self.merger_dol_cost += merged["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
                self.merger_dol_cost += merged["usage"]["completion_tokens"] * self.llmprice[1] / 1000

                red(f"#####\nTotal cost so far: ${self.merger_dol_cost:04f} ({merged['usage']['completion_tokens']} tokens)")

                # write to temp file
                with open("temp.md", "w") as f:
                    f.write(merged_text)

        else:
            raise ValueError(self.merge_mode)

        # write to temp file
        with open("temp.md", "w") as f:
            f.write(merged_text)

        self.merged_text = merged_text

    def _md_to_mindmap(self, md, title=None):
        """create mindmap directly from markdown"""
        mm = "- ```mermaid\nmindmap\n"

        headers = [
                blck
                for blck in md.splitlines()
                if blck.lstrip().startswith("- # ")
                ]
        if headers:
            first_header = headers[0][4:].replace("(", "").replace(")", "").strip()
            first_header = re.sub("\[nid\]nid:\d+", "", first_header).strip()
        else:
            first_header = ""

        md_clean = re.sub("- #+ ", "- ", md)
        md_clean = re.sub(mm_id_regex, "", md_clean)
        md_clean = re.sub("- ", "", md_clean)
        md_clean = md_clean.replace("(", "")
        md_clean = md_clean.replace(")", "")
        md_clean = re.sub("\[nid\]nid:\d+", "", md_clean).strip()
        md_clean = "\n".join([
            s.rstrip()
            for s in md_clean.splitlines()
            if s.strip() != "- ---"
            ]).strip()
        md_clean = dedent(md_clean)

        # mark the first header as root of the mindmap
        if title is None or levratio(title.lower(), first_header.lower()) > 90:
            md_clean = indent(md_clean, "\t").replace(f"\t{first_header}", f"root({first_header})", 1)
        else:
            md_clean = f"root({title.title()})\n" + indent(md_clean, "\t")

        mm += indent(md_clean, "\t")
        mm += "\n```"

        return mm

    def _expand_topics(self):
        """ask LLM to create a list of new topics based on the cheatsheet"""
        if not self.expand_topics:
            return
        whi("Creating new topics")

        sysprompt = graph_topic_maker_prompt.replace("DEFAULT", "\n".join(self.topic_list))
        sysprompt = dedent(sysprompt).strip()
        messages = [
                {
                    "role": "system",
                    "content": sysprompt,
                    },
                {
                    "role": "user",
                    "content": dedent(re.sub(mm_id_regex, "", self.merged_text)).strip(),
                    },
                ]
        new_topics_answer = self.cached_llm(
                messages=messages,
                llmname=self.llmname
                )

        self.topic_token_cost += new_topics_answer["usage"]["total_tokens"]
        self.topic_dol_cost += new_topics_answer["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
        self.topic_dol_cost += new_topics_answer["usage"]["completion_tokens"] * self.llmprice[1] / 1000
        red(f"Topic creation cost: ${self.topic_dol_cost:04f}")
        new_topics = new_topics_answer["choices"][0]["message"]["content"].splitlines()
        new_topics = [nt.strip() for nt in new_topics if nt.strip()]
        assert new_topics, "Empty list of new topics"
        assert not any(tp in self.topic_list for tp in new_topics), (
            f"Duplicate topics: {new_topics}")
        self.topic_list.extend(new_topics)

    def _main_loop_topicgraph_maker(self):
        """create graphs and/or topics for each questions and store them at the end of
        the merged file"""
        red("List of topics to use:")
        for lt in default_graph_topic:
            red(f"* {lt}")
        if self.expand_topics:
            red(f"  # newly generated: #")
            for lt in self.topic_list:
                if lt not in default_graph_topic:
                    red(f"* {lt}")

        self.graphs = {}
        self.cheatsheets = {}

        lock = Lock()
        topic_content_maker_prompt = dedent(topic_content_maker_prompt.replace("LANGUAGE", self.output_lang)).strip()
        graph_maker_prompt = dedent(graph_maker_prompt.replace("LANGUAGE", self.output_lang)).strip()
        def generate_item(topic):
            "from a topic create a graph or a topic content"
            whi(f"Creating graphs and/or cheatsheet for topic: {topic}")
            self.graphs[topic] = None
            if self.create_graphs:
                whi("Creating graph")
                messages = [
                        {
                            "role": "system",
                            "content": graph_maker_prompt.replace("TOPIC", topic),
                            },
                        {
                            "role": "user",
                            "content": re.sub(mm_id_regex, "", self.merged_text),
                            }
                        ]
                graph_answer = self.cached_llm(
                        messages=messages,
                        llmname=self.llmname,
                        )

                new_graph = graph_answer["choices"][0]["message"]["content"].strip()
                if new_graph.startswith("mermaid"):
                    new_graph = new_graph.replace("mermaid", "", 1).strip()
                new_graph = "```mermaid\n" + indent(dedent(new_graph), "\t") + "\n```"
                new_graph = new_graph.replace("(", "").replace(")", "")
                new_graph = new_graph.replace(" TD", " LR")
                new_graph = new_graph.replace("[", "[<font size=50>")
                assert new_graph.count("```") == 2, new_graph


                with lock:
                    self.graph_token_cost += graph_answer["usage"]["total_tokens"]
                    self.graph_dol_cost += graph_answer["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
                    self.graph_dol_cost += graph_answer["usage"]["completion_tokens"] * self.llmprice[1] / 1000

                    self.graphs[topic] = new_graph
                yel("Graph:")
                red(new_graph)
                red(f"Graph creation cost so far: ${self.graph_dol_cost:04f} ({graph_answer['usage']['completion_tokens']} tokens)")

            self.cheatsheets[topic] = None
            if self.create_cheatsheets:
                whi("Creating topic content")
                messages = [
                        {
                            "role": "system",
                            "content": topic_content_maker_prompt.replace("TOPIC", topic),
                            },
                        {
                            "role": "user",
                            # "content": re.sub(mm_id_regex, "", self.merged_text),
                            "content": self.merged_text,
                            }
                        ]
                topic_answer = self.cached_llm(
                        messages=messages,
                        llmname=self.llmname,
                        )

                new_cheatsheet = topic_answer["choices"][0]["message"]["content"]
                found = re.findall(mm_id_regex, new_cheatsheet)
                found = sorted(list(set(found)))
                for match in found:
                    assert match in self.mm_id, f"{match} not part of mm_id"
                    matched = self.mm_id[match]
                    new_cheatsheet = new_cheatsheet.replace(f"_{match}_", f"[nid](nid:{matched})")

                new_cheatsheet = dedent(new_cheatsheet).strip().replace("    ", "\t").replace("* ", "- ")
                new_cheatsheet = "\n".join(new_cheatsheet.splitlines())

                with lock:
                    self.cheatsheet_token_cost += topic_answer["usage"]["total_tokens"]
                    self.cheatsheet_dol_cost += topic_answer["usage"]["prompt_tokens"] * self.llmprice[0] / 1000
                    self.cheatsheet_dol_cost += topic_answer["usage"]["completion_tokens"] * self.llmprice[1] / 1000
                    self.cheatsheets[topic] = new_cheatsheet
                yel("Topic content:")
                red(new_cheatsheet)
                red(f"Topic content creation cost so far: ${self.cheatsheet_dol_cost:04f} ({topic_answer['usage']['completion_tokens']} tokens)")

        Parallel(
                n_jobs=5,
                backend="threading")(
                        delayed(generate_item)(topic)
                        for topic in tqdm(
                            self.topic_list,
                            desc="Generating graph and cheatsheet",
                            )
                        )


    def _export_to_logseq(self):
        """write the final markdown to logseq"""
        whi(f"Writing to {self.output}")
        txt = ""

        # replace the node id
        merged_text = self.merged_text
        found = re.findall(mm_id_regex, merged_text)
        found = sorted(list(set(found)))
        hallucinated = []
        for match in found:
            if match not in self.mm_id:
                red(f"{match} nid was apparently hallucinated!")
                hallucinated.append(match)
            matched = self.mm_id[match]
            merged_text = merged_text.replace(f"_{match}_", f"[nid](nid:{matched})")

        # figure out how many nid are missing
        cnt_in = 0
        cnt_missing = 0
        missings = []
        missings_id2 = []
        for mmid in self.mm_id.values():
            if mmid in merged_text:
                cnt_in += 1
            else:
                cnt_missing += 1
                missings.append(mmid)
                missings_id2.append([k for k,v in self.mm_id.items() if v == mmid][0])
        ratiomissing = cnt_missing / (cnt_missing + cnt_in) * 100

        toptag = ",".join(self.toptags)
        txt += f"AnkiAutoMindmap_toptag:: {toptag}\n"
        d = datetime.today()
        today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"
        txt += f"AnkiAutoMindmap_date:: {today}\n"
        txt += f"AnkiAutoMindmap_version:: {self.VERSION}\n"
        total_token_cost = sum(
                [
                    self.graph_token_cost,
                    self.cheatsheet_token_cost,
                    self.topic_token_cost,
                    self.merger_token_cost,
                    ])
        total_dol_cost = sum(
                [
                    self.graph_dol_cost,
                    self.cheatsheet_dol_cost,
                    self.topic_dol_cost,
                    self.merger_dol_cost,
                    ])
        txt += f"AnkiAutoMindmap_total_token_cost:: {total_token_cost}\n"
        txt += f"AnkiAutoMindmap_total_dol_cost:: {total_dol_cost:04f}\n"
        txt += f"AnkiAutoMindmap_graph_dol_cost:: {self.graph_dol_cost:04f}\n"
        txt += f"AnkiAutoMindmap_cheatsheet_dol_cost:: {self.cheatsheet_dol_cost:04f}\n"
        txt += f"AnkiAutoMindmap_merger_dol_cost:: {self.merger_dol_cost:04f}\n"
        txt += f"AnkiAutoMindmap_topic_dol_cost:: {self.topic_dol_cost:04f}\n"
        txt += f"AnkiAutoMindmap_LLm_used:: {self.llmname}\n"
        txt += f"AnkiAutoMindmap_nb_cards:: {cnt_in}\n"
        txt += f"AnkiAutoMindmap_nb_missing_cards:: {cnt_missing}\n"
        txt += f"AnkiAutoMindmap_ratio_missing:: {ratiomissing:02f}\n"
        txt += f"AnkiAutoMindmap_n_hallucinated:: {len(hallucinated)}\n"
        txt += "\n\n"
        if missings:
            txt += f"- List of missing nids\n"
            txt += f"\t- nid:{','.join(missings)}\n"
            txt += f"\t- temp_nid:{','.join(missings_id2)}\n"
        if hallucinated:
            txt += f"- List of hallucinated note identifiers\n\t- {','.join(hallucinated)}\n"

        txt += merged_text.strip()
        if self.create_mindmaps:
            txt += "\n- # Mindmap {{renderer :mermaid_" + self.hash(self.merged_text) + "}}\n"
            txt += "  collapsed:: true\n"
            txt += indent(self._md_to_mindmap(self.merged_text), "\t") + "\n"

        if self.create_graphs or self.create_cheatsheets:
            txt += "\n- ---"
            txt += "\n- ---"
            txt += "\n- # Cheatsheets\n"
            for topic in self.graphs.keys():
                graph = self.graphs[topic]
                cheatsheet = self.cheatsheets[topic]

                txt += f"\n\t- ## {topic}\n"
                txt += "\t  collapsed:: true\n"

                if cheatsheet is not None:
                    chsh_lines = dedent(cheatsheet).splitlines()
                    chsh_lines = [c for c in chsh_lines if c.strip()]
                    for i, chsh in enumerate(chsh_lines):

                        # remove headers
                        # chsh = re.sub("- #+ ", "- ", chsh)

                        if not chsh.strip().startswith("- "):
                            first_char = chsh.strip()[0]
                            chsh = chsh.replace(first_char, f"- {first_char}", 1)
                        if i == 0:
                            txt += "\n\t\t" + chsh + "\n"
                        else:
                            txt += indent(chsh, "\t\t") + "\n"

                    if self.create_mindmaps:
                        txt += "\n\t\t- ### Mindmap {{renderer :mermaid_" + self.hash(cheatsheet) + "}}\n"
                        txt += "\t\t  collapsed:: true\n"
                        txt += indent(self._md_to_mindmap(cheatsheet, topic), "\t\t\t") + "\n"

                if graph is not None:
                    txt += "\t- {{renderer :mermaid_" + self.hash(graph) + "}}\n"
                    graph_lines = dedent(graph).splitlines()
                    graph_lines = [g for g in graph_lines if g.strip()]
                    for i, grl in enumerate(graph_lines):
                        if i == 0:
                            if not grl.strip().startswith("- "):
                                first_char = grl.strip()[0]
                                grl = grl.replace(first_char, f"- {first_char}", 1)
                            txt += "\n\t\t\t" + grl + "\n"
                        else:
                            txt += indent(grl, "\t\t\t  ") + "\n"
                    txt += "\n\t- ---"

        with open(self.output, "w") as f:
            f.write(txt)
        whi(f"Done writing to {self.output}")

        red("\n\nFinal text ######")
        whi(txt)

    def format_patch(self, patch):
        assert isinstance(patch, str)
        patch = patch.strip()

        # remove wrapper if present
        if patch.startswith("```"):
            patch = "\n".join(patch.splitlines()[1:])
        if patch.endswith("```"):
            patch = "\n".join(patch.splitlines()[:-1])

        lines = patch.splitlines()
        paths = lines[:2]
        if all(p.startswith("---") or p.startswith("+++") for p in paths):
            assert paths[0].startswith("---"), f"First line doesn't start with ---: {paths[0]}"
            assert paths[1].startswith("+++"), f"Second line doesn't start with +++: {paths[0]}"
            lines[0] = "--- cheatsheet1.md"
            lines[1] = "+++ cheatsheet1.md"
        else:
            lines.insert(0, "+++ cheatsheet1.md")
            lines.insert(0, "--- cheatsheet1.md")

        # remove line number if present
        for i, l in enumerate(lines):
            while l[0].isdigit():
                l = l[1:]
            lines[i] = l

        patch = "\n".join(lines)
        return patch



if __name__ == "__main__":
    intance = fire.Fire(AnkiAutoMindmap)
