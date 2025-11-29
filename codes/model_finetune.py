from datasets import load_dataset, ClassLabel, Features, Value
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score


train_path = ""  # Path your train dataset
test_path = ""  # Path your test dataset
SAVE_DIR = ""  # Path direction that model will save in
LOCAL_MODEL_DIR = "" # Path direction that model save in


dataset = load_dataset(
    'csv',
    data_files={
        'train': train_path,
        'test': test_path
    },
    # delimiter="\t",      # ***Unmark this line if your dataset is with *tsv* format***
)

label_names = ["HAPPY", "SAD", "FEAR", "ANGRY", "HATE", "SURPRISE", "OTHER"]
features = Features({'text': Value('string'), 'label': ClassLabel(names=label_names)})
dataset = dataset.cast(features)

split = dataset['train'].train_test_split(test_size=0.1, stratify_by_column='label', seed=42)
dataset = {'train': split['train'], 'validation': split['test'], 'test': dataset['test']}

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, max_length=256)

tokenized = {}
for split_names in ['train', 'validation', 'test']:
    tokenized[split_names] = dataset[split_names].map(
        tokenize_fn, batched=True, remove_columns=['text']
    )

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

id2label = {i: l for i, l in enumerate(label_names)}
label2id = {l: i for i, l in enumerate(label_names)}

model = AutoModelForSequenceClassification.from_pretrained(
    LOCAL_MODEL_DIR,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
    local_files_only=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

args = TrainingArguments(
    output_dir="fabert-sentiment",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    logging_steps=50,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()
eval_test = trainer.evaluate(tokenized["test"])
print(eval_test)

best_ckpt = trainer.state.best_model_checkpoint
print("Best checkpoint:", best_ckpt)

save_dir = SAVE_DIR

best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
best_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)