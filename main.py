import numpy as np
from llama_cpp import (
    Llama,
    llama_get_logits,
    llama_chat_apply_template,
    llama_chat_format,
    llama_kv_cache_clear,
)
import torch
import torch.nn.functional as F
import sys
import random
from typing import List, Tuple
from sag import finish_last_equation

# Load the model
model = Llama(
    "/Users/hudsongouge/.ollama/models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
    n_gpu_layers=-1,
    n_ctx=512,
    verbose=False,
)

tokenizer = model.tokenizer()


# [
#     print(tokenizer.decode([token]), end="", flush=True)
#     for token in model.generate(tokenizer.encode("Hello, my name is"), temp=0)
# ]


sequence_breaker_strings = ['"\\n", ":", "\\"", "*"']
sequence_breakers = {tokenizer.encode(f"a{s}")[-1] for s in sequence_breaker_strings}
# print(llama_chat_format.format_qwen(messages=[{"role": "user", "content": "hello"}]))


def calculate_dry_penalty(
    input_ids: torch.LongTensor,
    scores: torch.FloatTensor,
    _range: int = 1024,
    sequence_breakers=sequence_breakers,
    allowed_length=1,
    base=2,
    multiplier=3,
) -> torch.FloatTensor:
    if _range > 0:
        input_ids = input_ids[:, -_range:]

    for input_ids_row, scores_row in zip(input_ids, scores):
        # Raw integer must be extracted here to check for set membership.
        last_token = input_ids_row[-1].item()

        if last_token in sequence_breakers:
            continue

        # Exclude the last token as it always matches.
        match_indices = (input_ids_row[:-1] == last_token).nonzero()

        # Stores the maximum matching sequence length
        # for each token immediately following the sequence in the input.
        match_lengths = {}

        for i in match_indices:
            next_token = input_ids_row[i + 1].item()

            if next_token in sequence_breakers:
                continue

            # We have already found that `last_token` matches at this index,
            # so the match is at least of length 1.
            match_length = 1

            # Extend the match backwards as far as possible.
            while True:
                j = i - match_length
                if j < 0:
                    # Start of input reached.
                    break

                previous_token = input_ids_row[-(match_length + 1)].item()
                if input_ids_row[j] != previous_token:
                    # Start of match reached.
                    break

                if previous_token in sequence_breakers:
                    # Sequence-breaking token reached.
                    break

                match_length += 1

            if next_token in match_lengths:
                match_lengths[next_token] = max(match_length, match_lengths[next_token])
            else:
                match_lengths[next_token] = match_length

        # Apply penalties.
        for token, match_length in match_lengths.items():
            if match_length >= allowed_length:
                penalty = multiplier * base ** (match_length - allowed_length)
                scores_row[token] -= penalty

    return scores


def inference(tokens, model):
    # try:
    model.eval(tokens)
    # except:
    #     print("\x1b[31m", "error:", tokenizer.decode(tokens), ":\x1b[0m")
    #     sys.exit(1)

    logits_ptr = llama_get_logits(model.ctx)
    next_token_logits = torch.tensor(
        np.array([np.ctypeslib.as_array(logits_ptr, shape=(1, model.n_vocab()))[-1]])
    )

    return next_token_logits


def generate_top_k_token_with_prob(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1,
    min_prob: float = 0,
    randomness: bool = True,
) -> List[Tuple[torch.Tensor, float]]:
    scaled_logits = logits / temperature
    probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
    probabilities = torch.where(
        probabilities > min_prob, probabilities, torch.tensor(0.0)
    )

    if randomness:
        # Sample from the filtered probabilities
        samples = torch.multinomial(probabilities, num_samples=k)
        selected_probabilities = [prob.item() for prob in probabilities[0, samples][0]]
    else:
        # Get the top k indices and their probabilities
        samples = torch.topk(probabilities, k, dim=-1).indices
        selected_probabilities = torch.topk(probabilities, k, dim=-1).values

    top_k_with_prob = sorted(
        zip(samples[0], [prob for prob in selected_probabilities[0] if prob > 0]),
        key=lambda x: x[1],
        reverse=True,
    )
    return top_k_with_prob


def generate(
    tokens,
    k=3,
    depth=5,
    max_new_tokens=100,
    deep=0,
    temperature=1.0,
    min_prob=0,
):
    all_tokens = tokens
    all_tokens_text = tokenizer.decode(all_tokens)
    tokens = torch.tensor([tokens])
    num_original_tokens = len(all_tokens)
    confidence = 0
    while len(all_tokens) - num_original_tokens < max_new_tokens:
        finished = finish_last_equation(all_tokens_text)
        if finished:
            new_tokens = tokenizer.encode(finished.removeprefix(all_tokens_text))
            new_tokens_tensor = torch.tensor([new_tokens])
            tokens = torch.cat((tokens, new_tokens_tensor), dim=1)
            all_tokens_text = finished
            all_tokens += new_tokens
        # Get logits from the model
        try:
            logits = inference(tokens[0], model)
        except:
            print(tokens, deep)
            sys.exit(1)

        post_dry = calculate_dry_penalty(tokens, logits)

        top_k_with_prob = generate_top_k_token_with_prob(
            post_dry, k, temperature, min_prob, False
        )
        chosen_token = None
        if top_k_with_prob[0][1] > sum([x[1] for x in top_k_with_prob[1:]]):
            chosen_token = top_k_with_prob[0][0]
            tokens = chosen_token.view(1, 1)
            all_tokens.append(chosen_token)
            all_tokens_text += tokenizer.decode([chosen_token])
            confidence += top_k_with_prob[0][1]
        elif depth > 1 and deep < 2:
            best_confidence = -1
            best_tokens = all_tokens
            for tok, prob in top_k_with_prob:
                llama_kv_cache_clear(model.ctx)
                generated_tokens, generation_confidence = generate(
                    all_tokens + [tok],
                    k=k,
                    depth=depth // 2,
                    max_new_tokens=depth,
                    deep=deep + 1,
                )
                # print(
                #     "-" * deep
                #     + tokenizer.decode(generated_tokens).replace("\n", "\\n"),
                #     float(generation_confidence),
                # )
                if generation_confidence > best_confidence:
                    best_confidence = generation_confidence
                    best_tokens = generated_tokens
            llama_kv_cache_clear(model.ctx)
            # print("best-tokens", best_tokens)
            tokens = torch.tensor([best_tokens])
            all_tokens = best_tokens
            all_tokens_text = tokenizer.decode(all_tokens)
            confidence += best_confidence
        else:
            chosen_token = top_k_with_prob[0][0]
            tokens = chosen_token.view(1, 1)
            all_tokens.append(chosen_token)
            all_tokens_text += tokenizer.decode([chosen_token])
            confidence += top_k_with_prob[0][1]
        if deep == 0:
            print(all_tokens_text, "\nNEW:")
    return all_tokens, confidence


print(tokenizer.decode(generate(tokenizer.encode("1+1="))[0]))
