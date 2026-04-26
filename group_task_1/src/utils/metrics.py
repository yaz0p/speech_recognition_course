import editdistance
import torch

BLANK_ID = 10  # CTC blank token; must match vocab_size - 1 in config


def greedy_decoder(
    log_probs: torch.Tensor, blank_id: int = BLANK_ID
) -> list[list[int]]:
    """Greedy CTC decoder: argmax → collapse repeats → remove blanks."""
    preds = torch.argmax(log_probs, dim=-1)

    decoded_batch = []
    for batch_idx in range(preds.shape[0]):
        decoded_seq = []
        previous_char = None
        for char in preds[batch_idx].tolist():
            if char not in {previous_char, blank_id}:
                decoded_seq.append(char)
            previous_char = char
        decoded_batch.append(decoded_seq)

    return decoded_batch


def compute_cer(predictions: list[list[int]], targets: list[list[int]]) -> float:
    """Character Error Rate via Levenshtein distance on token lists."""
    total_distance = 0
    total_length = 0
    for pred, target in zip(predictions, targets, strict=False):
        total_distance += editdistance.eval(pred, target)
        total_length += len(target)

    return total_distance / total_length if total_length > 0 else 0.0
