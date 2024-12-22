from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_task_model(task_name, training_data):
    # Define model and load dataset
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    dataset = load_dataset("json", data_files={"train": training_data})

    train_dataset = dataset['train']

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{task_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Create and train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()

    return {"message": f"Fine-tuning completed for task: {task_name}"}
