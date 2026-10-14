"""Shared data, generation and retrieval for the 3 notebooks

The BRIGHT mapping/exclusions and Qwen loading follow R5
Use SentenceTransformer directly to avoid R5's unrelated setup.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# using R5 code
def load_biology(revision):
    from datasets import load_dataset

    # Read files directly so BRIGHT's metadata cannot require the other domain splits.
    base = f'hf://datasets/xlangai/BRIGHT@{revision}'
    examples = load_dataset('parquet', data_files=f'{base}/examples/biology-*.parquet', split='train')
    documents = load_dataset('parquet', data_files=f'{base}/documents/biology-*.parquet', split='train')
    queries = examples.to_pandas()[['id', 'query', 'gold_ids', 'excluded_ids']].copy()
    queries['qid'] = 'biology/' + queries.id.astype(str)
    corpus = documents.to_pandas().rename(columns={'id': 'docno', 'content': 'text'})
    corpus['docno'] = corpus.docno.astype(str)
    queries['gold_ids'] = queries.gold_ids.map(lambda ids: list(map(str, ids)))
    queries['excluded_ids'] = queries.excluded_ids.map(lambda ids: set(map(str, ids)) - {'N/A'})
    assert queries.qid.is_unique and corpus.docno.is_unique
    docnos = set(corpus.docno)
    for row in queries.itertuples():
        assert row.gold_ids and set(row.gold_ids) <= docnos
        assert not set(row.gold_ids) & row.excluded_ids
    qrels = pd.DataFrame([{'qid': row.qid, 'docno': docno, 'label': 1}
                          for row in queries.itertuples() for docno in row.gold_ids])
    return queries, corpus, qrels


def split_response(raw):
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Incomplete response. Inspect the cache; revise the budget in a new run.')
    message = choice['message']
    answer = message.get('content') or ''
    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    if not reasoning and '</think>' in answer:
        reasoning, answer = answer.split('</think>', 1)
        reasoning = reasoning.split('<think>', 1)[-1]
    if not isinstance(reasoning, str) or not reasoning.strip() or not answer.strip():
        raise ValueError('The service must expose both reasoning and a final answer.')
    return dict(reasoning=reasoning.strip(), answer=answer.strip(),
                full_text=reasoning.strip() + '\n\n' + answer.strip())


def read_traces(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    latest = {record['qid']: record for record in records}
    return pd.DataFrame([dict(qid=qid, **split_response(record['raw']),
                              seconds=record['seconds'], usage=record['raw'].get('usage'))
                         for qid, record in latest.items()])


def generate(queries, prompt, settings, path, client):
    path = Path(path)
    generation_settings = {key: settings[key] for key in
                           ['model', 'base_url', 'max_new_tokens', 'temperature', 'top_p'] if key in settings}
    saved = {record['qid']: record for record in
             (json.loads(line) for line in path.read_text().splitlines() if line.strip())} if path.exists() else {}
    for row in queries.itertuples():
        text = prompt.format(query=row.query)
        if row.qid in saved:
            record = saved[row.qid]
            if record['prompt'] != text or record['settings'] != generation_settings:
                raise ValueError('Cached generation settings changed. Choose a new output folder/file.')
            split_response(record['raw'])
            continue
        print('Generating', row.qid, flush=True)
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=settings['model'], messages=[{'role': 'user', 'content': text}],
            max_tokens=settings['max_new_tokens'], temperature=settings['temperature'],
            top_p=settings['top_p'], timeout=300)
        record = dict(qid=row.qid, prompt=text, settings=generation_settings,
                      seconds=time.perf_counter() - started, raw=response.model_dump(mode='json'))
        with path.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        split_response(record['raw'])  # Save failed responses too, then stop before evaluation.
    return read_traces(path).set_index('qid').loc[queries.qid].reset_index()


def token_count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def prefix(text, budget, tokenizer):
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    if budget <= 0 or not offsets:
        return ''
    if budget >= len(offsets):
        return text
    end = offsets[budget - 1][1]
    while end and token_count(text[:end], tokenizer) > budget:
        end -= 1
    return text[:end]


# using R5 code
def eligible_top_ten(run, excluded):
    run = run.copy()
    run['qid'] = run.qid.astype(str)
    run['docno'] = run.docno.astype(str)
    keep = [docno not in excluded[qid] for qid, docno in zip(run.qid, run.docno)]
    run = run.loc[keep].sort_values(['qid', 'score', 'docno'], ascending=[True, False, True])
    run = run.groupby('qid', sort=False).head(10).copy()
    run['rank'] = run.groupby('qid', sort=False).cumcount()
    return run


# using R5 code
def make_bm25(corpus, index_dir, depth, controls):
    import pyterrier as pt

    index_dir = Path(index_dir).resolve()
    index = pt.terrier.TerrierIndex(str(index_dir))
    if not (index_dir / 'data.properties').exists():
        index.indexer(meta={'docno': int(corpus.docno.str.len().max())}).index(
            corpus[['docno', 'text']].to_dict('records'))
    return pt.rewrite.tokenise() >> pt.terrier.Retriever(index, wmodel='BM25',
                                                       controls=dict(controls), num_results=depth)


# using R5 code
def load_encoder(config):
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime for Qwen3-Embedding-4B.')
    model = SentenceTransformer(config['embedding_model'], device='cuda',
        model_kwargs={'dtype': torch.float16, 'attn_implementation': 'sdpa'},
        tokenizer_kwargs={'padding_side': 'left'})
    model[0].auto_model.config.use_cache = False  # Embedding whole texts does not need generation KV caches.
    model.max_seq_length = config['max_length']
    print('Embedding query instruction:', model.prompts['query'])
    return model


# using R5 code
def encode_texts(model, texts, batch_size, query=False):
    # Query instructions follow Qwen/R5. Never silently truncate a query variant.
    instruction = model.prompts.get('query', '') if query else ''
    lengths = [len(model.tokenizer.encode(instruction + text)) for text in texts]
    if max(lengths, default=0) > model.max_seq_length:
        raise ValueError('Embedding context is too short. Increase max_length or declare a truncation experiment.')
    embeddings = model.encode(texts, batch_size=batch_size,
                              prompt_name='query' if query else None, prompt='' if not query else None,
                              normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    assert np.isfinite(embeddings).all()
    return embeddings


def corpus_embeddings(model, corpus, config, path):
    identity = {key: config[key] for key in ['dataset_revision', 'embedding_model', 'max_length']}
    identity['dimension'] = model.get_sentence_embedding_dimension()
    identity['model_revision'] = getattr(model[0].auto_model.config, '_commit_hash', None)
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            assert json.loads(str(cached['identity'])) == identity, 'Use a new embedding cache.'
            assert np.array_equal(cached['docnos'], corpus.docno.to_numpy())
            embeddings = cached['embeddings']
        assert embeddings.shape == (len(corpus), identity['dimension']) and np.isfinite(embeddings).all()
        return embeddings
    embeddings = encode_texts(model, corpus.text.tolist(), config['doc_batch_size'])
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, identity=json.dumps(identity), docnos=corpus.docno.to_numpy(dtype=str), embeddings=embeddings)
    temporary.replace(path)
    return embeddings


def dense_search(topics, model, embeddings, docnos, batch_size, depth):
    queries = encode_texts(model, topics['query'].tolist(), batch_size, query=True)
    scores = queries @ embeddings.T  # Exact search over the full passage corpus, on CPU.
    docnos = np.asarray(docnos, dtype=str)
    depth = min(depth, len(docnos))
    rows = []
    for row, values in zip(topics.itertuples(), scores):
        threshold = np.partition(values, len(values) - depth)[len(values) - depth]
        candidates = np.flatnonzero(values >= threshold)  # Include boundary ties before sorting.
        order = candidates[np.lexsort((docnos[candidates], -values[candidates]))][:depth]
        rows.extend(dict(qid=row.qid, docno=docnos[i], score=float(values[i]), rank=rank)
                    for rank, i in enumerate(order))
    return pd.DataFrame(rows, columns=['qid', 'docno', 'score', 'rank'])


def retrievers(corpus, queries, config, root, use_dense=False):
    excluded = dict(zip(queries.qid, queries.excluded_ids))
    depth = 10 + max(map(len, excluded.values()), default=0)
    bm25 = make_bm25(corpus, Path('/content/biology_indexes') / config['dataset_revision'],
                    depth, config['bm25_controls'])
    systems = {'BM25': bm25.transform}
    if not use_dense:
        return systems, None
    model = load_encoder(config)
    embeddings = corpus_embeddings(model, corpus, config, Path(root) / 'corpus_embeddings.npz')
    systems['Qwen3'] = lambda topics: dense_search(topics, model, embeddings, corpus.docno,
                                                  config['query_batch_size'], depth)
    return systems, model


def evaluate(variants, systems, queries, qrels):
    import pyterrier as pt

    assert not variants.duplicated(['qid', 'variant']).any()
    assert variants.text.map(lambda text: isinstance(text, str) and bool(text.strip())).all()
    expected = set(queries.qid)
    excluded = dict(zip(queries.qid, queries.excluded_ids))
    runs = {}
    for name, group in variants.groupby('variant', sort=False):
        assert set(group.qid) == expected, 'Every variant must cover the same queries.'
        topics = group[['qid', 'text']].rename(columns={'text': 'query'})
        for model, retrieve in systems.items():
            runs[f'{model}/{name}'] = eligible_top_ten(retrieve(topics), excluded)
    means, per_query = pt.Experiment(list(runs.values()), queries[['qid', 'query']], qrels,
        names=list(runs), eval_metrics=[pt.measures.nDCG @ 10], perquery='both')
    assert len(per_query) == len(runs) * len(queries) and per_query.value.notna().all()
    ranks = pd.concat([run.assign(name=name) for name, run in runs.items()], ignore_index=True)
    return means, per_query, ranks


def take_sentences(text, spans, order, budget, tokenizer):
    selected = []
    for i in order:
        candidate = sorted(selected + [int(i)])
        shortened = '\n'.join(text[spans[j][0]:spans[j][1]].strip() for j in candidate)
        if token_count(shortened, tokenizer) > budget:
            break  # Keep a prefix of the ranking; budget handling does not change MMR's equation.
        selected = candidate
    if not selected:
        raise ValueError('The first ranked sentence exceeds the budget. Increase the budget for this case.')
    return '\n'.join(text[spans[i][0]:spans[i][1]].strip() for i in selected), selected
