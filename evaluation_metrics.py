import evaluate  # Hugging Face's evaluate library
import numpy as np
import torch
from bert_score import BERTScorer, score as bert_score



# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def compute_metrics_for_pretrain(eval_preds, tokenizer):
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



bert_scorer = BERTScorer(
    lang="en",
    model_type="bert-base-uncased",
    device="cuda" if torch.cuda.is_available() else "cpu",
    idf=False,  # Disable IDF to save memory
    rescale_with_baseline=True  # Better score normalization
)


def compute_metrics_for_qna(eval_preds, tokenizer):
    """ Metric computation for qna finetune """
    with torch.no_grad():
        logits, labels = eval_preds
        
        # Convert to numpy (move to CPU first if needed)
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        
        # Get predicted tokens (shape: [batch_size, seq_length])
        pred_ids = np.argmax(logits, axis=-1)
        
        # Decode in batches to avoid memory spikes
        batch_size = 8  # Adjust based on your GPU memory
        pred_str, label_str = [], []
        
        for i in range(0, len(pred_ids), batch_size):
            # Decode predictions
            batch_preds = pred_ids[i:i+batch_size]
            pred_str.extend(tokenizer.batch_decode(
                batch_preds, 
                skip_special_tokens=True
            ))
            
            # Decode labels (replace -100 with pad_token_id)
            batch_labels = labels[i:i+batch_size]
            batch_labels = np.where(
                batch_labels != -100, 
                batch_labels, 
                tokenizer.pad_token_id
            )
            label_str.extend(tokenizer.batch_decode(
                batch_labels, 
                skip_special_tokens=True
            ))
        
        # Skip if empty (avoid errors)
        if not pred_str or not label_str:
            return {
                'bleu': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bertscore_f1': 0.0
            }
        
        # Compute BLEU (handle edge cases)
        try:
            bleu_score = bleu_metric.compute(
                predictions=pred_str,
                references=[[ref] for ref in label_str]
            )['bleu']
        except:
            bleu_score = 0.0
        
        # Compute ROUGE
        rouge_scores = rouge_metric.compute(
            predictions=pred_str,
            references=label_str,
            use_stemmer=True
        )
        
        # Compute BERTScore in batches
        P, R, F1 = bert_scorer.score(
            pred_str, 
            label_str,
            batch_size=4  # Small batch for BERTScore
        )
        
        metrics = {
            'bleu': bleu_score,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item(),
        }
        
        torch.cuda.empty_cache()
        return metrics