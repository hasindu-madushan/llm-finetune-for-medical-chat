import evaluate  # Hugging Face's evaluate library
import numpy as np
import torch


# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def compute_metrics_for_pretrain(eval_preds):
    """ Calculate bleu and rouge scores """
    with torch.no_grad():
        logits, labels = eval_preds
        
        logits = logits.cpu().numpy() if torch.is_tensor(logits) else logits
        
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        # Get predicted token IDs (argmax of logits)
        pred_ids = np.argmax(logits, axis=-1)  # Shape: (batch_size, seq_length)
        
        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
        # Replace -100 with pad_token_id in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        references = [[ref] for ref in label_str]

        
        # Compute BLEU
        bleu_result = bleu_metric.compute(
            predictions=pred_str,
            references=references
        )
        
        # Compute ROUGE
        rouge_result = rouge_metric.compute(
            predictions=pred_str,
            references=label_str,
            use_stemmer=True
        )
        
        # Extract main scores
        metrics = {
            'bleu': bleu_result['bleu'],
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeL'],
        }
        torch.cuda.empty_cache()
        
        return metrics


