from mlx_lm.generate import BatchGenerator

from RAG_Framework.core.BPE_decode import BPEDecoder


def run_batch_generate(llm_model, llm_tokenizer, prompts, max_tokens,
                        label="BATCH", on_complete=None, sampler=None):
    """Run BatchGenerator for a list of prompts simultaneously.

    Returns a list of decoded strings in the same order as *prompts*.
    Falls back to sequential generate_fixed() for a single prompt.

    on_complete(idx, result): optional callback fired immediately when each
    individual sequence finishes, before the full batch completes.
    """
    from RAG_Framework.agents.planner import generate_fixed

    if not prompts:
        return []

    if len(prompts) == 1:
        result = generate_fixed(
            llm_model, llm_tokenizer,
            prompt=prompts[0], max_tokens=max_tokens, verbose=True,
            sampler=sampler
        )
        if on_complete:
            on_complete(0, result)
        return [result]

    print(f"\n[{label}] BatchGenerator: {len(prompts)} prompts, max_tokens={max_tokens}")

    token_prompts = []
    for p in prompts:
        if isinstance(p, str):
            token_prompts.append(llm_tokenizer.encode(p, add_special_tokens=False))
        else:
            token_prompts.append(p)

    batch_kwargs = dict(max_tokens=max_tokens)
    if sampler is not None:
        batch_kwargs["sampler"] = sampler

    batch_gen = BatchGenerator(llm_model, **batch_kwargs)
    uids = batch_gen.insert(token_prompts)
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}
    uid_to_token_ids = {uid: [] for uid in uids}
    active = set(uids)
    results = [None] * len(prompts)

    while active:
        for resp in batch_gen.next():
            if resp.uid in active:
                uid_to_token_ids[resp.uid].append(resp.token)
                if resp.finish_reason:
                    active.remove(resp.uid)
                    idx = uid_to_idx[resp.uid]
                    text = BPEDecoder.decode_tokens(llm_tokenizer, uid_to_token_ids[resp.uid])
                    results[idx] = text
                    print(f"[{label}] Sequence {idx} done ({len(text)} chars)")
                    if on_complete:
                        on_complete(idx, text)

    batch_gen.close()
    print(f"[{label}] BatchGenerator complete")
    return results
