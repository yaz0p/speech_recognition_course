import torch
from typing import List

def greedy_decoder(log_probs: torch.Tensor, blank_id: int = 10) -> List[List[int]]:
    """
    Greedy CTC decoder.
    Takes the argmax at each time step, collapses repeated characters,
    and removes blanks.
    
    Args:
        log_probs: Tensor of shape [batch, time, vocab_size]
        blank_id: ID of the blank token
        
    Returns:
        List of lists containing decoded sequences
    """
    # Get argmax over vocab dimension
    # shape: [batch, time]
    preds = torch.argmax(log_probs, dim=-1)
    
    decoded_batch = []
    for batch_idx in range(preds.shape[0]):
        pred_sequence = preds[batch_idx].tolist()
        
        decoded_seq = []
        previous_char = None
        for char in pred_sequence:
            # Collapse repeated characters
            if char != previous_char:
                # Remove blanks
                if char != blank_id:
                    decoded_seq.append(char)
            previous_char = char
            
        decoded_batch.append(decoded_seq)
        
    return decoded_batch

def compute_cer(predictions: List[List[int]], targets: List[List[int]]) -> float:
    """
    Compute Character Error Rate.
    We convert the sequences back to strings to use string edit distance or logic.
    For digits, Levenshtein distance on sequences works perfectly.
    """
    import editdistance
    
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        # Calculate edit distance between lists directly
        distance = editdistance.eval(pred, target)
        total_distance += distance
        total_length += len(target)
        
    if total_length == 0:
        return 0.0
        
    return total_distance / total_length
