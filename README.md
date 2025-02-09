# Sentiment Analysis with Deep Learning using BERT

## Overview
This project demonstrates how to perform sentiment analysis using BERT, a powerful transformer-based language model. The implementation follows a step-by-step approach, covering data preprocessing, model training, and evaluation. The model is trained on a labeled dataset using full fine-tuning, meaning all transformer layers are updated during training.

## Learning Objectives
you will learn how to analyze a dataset for sentiment analysis. You will learn how to read in a PyTorch BERT model, and adjust the architecture for multi-class classification. You will learn how to adjust an optimizer and scheduler for ideal training and performance. In finetuning this model, you will learn how to design a train and evaluate loop to monitor model performance as it trains, including saving and loading models. Finally, you will build a Sentiment Analysis model that leverages BERT's large-scale language knowledge.

- What BERT is and what it can do
- Clean and preprocess text dataset
- Split dataset into training and validation sets using stratified approach
- Tokenize (encode) dataset using BERT toknizer
- Design BERT finetuning architecture
- Evaluate performance using F1 scores and accuracy
- Finetune BERT using training loop

## Fine-Tuning Method
The project employs full fine-tuning of a BERT-based model (BertForSequenceClassification). This method updates all parameters, including transformer layers, rather than only modifying the classification head. Key elements of fine-tuning in this project:
- Model Used: BertForSequenceClassification
- Optimizer: AdamW (optimized for transformers)
- Gradient Updates: loss.backward() and optimizer.step() ensure all model parameters are updated.
- Learning Rate Scheduler: A scheduler is used to adjust learning rates dynamically during training.

## Prerequisites
To follow along with this project, you should have:
- Intermediate-level knowledge of Python 3 (NumPy and Pandas preferable but not required)
- Basic exposure to PyTorch
- Understanding of Deep Learning concepts and Language Models (specifically BERT)

## Project Outline
1. **Introduction** - Overview of BERT and project objectives.
2. **Exploratory Data Analysis and Preprocessing** - Loading and cleaning the dataset.
3. **Training/Validation Split** - Splitting the dataset for training and validation.
4. **Loading Tokenizer and Encoding Data** - Tokenizing text and preparing input tensors.
5. **Setting up BERT Pretrained Model** - Loading a pretrained BERT model for classification.
6. **Creating Data Loaders** - Preparing data for training in batches.
7. **Setting Up Optimizer and Scheduler** - Configuring optimization strategies.
8. **Defining Performance Metrics** - Implementing F1-score and accuracy measurements.
9. **Creating Training Loop** - Implementing the training process.
10. **Loading and Evaluating Model** - Testing the model with real data.

## Dataset
We use the **SMILE Twitter Emotion dataset**:
> Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016): SMILE Twitter Emotion dataset. figshare. Dataset. [https://doi.org/10.6084/m9.figshare.3187909.v2](https://doi.org/10.6084/m9.figshare.3187909.v2)

### Data Processing
- Filtering out multi-label entries and "nocode" labels
- Encoding categorical labels into numerical values
- Splitting data into training and validation sets

## Model Architecture
- **BERT** (Bidirectional Encoder Representations from Transformers)
- Fine-tuned for sentiment classification with **six categories**: `happy`, `not-relevant`, `angry`, `disgust`, `sad`, `surprise`.

## Implementation Steps
### 1. Tokenization and Encoding
Using `BertTokenizer` from HuggingFace to preprocess text inputs.
```python
from transformers import BertTokenizer

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Encode training and validation data
encoded_data_train = tokenizer.batch_encode_plus(df[df.data_type == 'train'].text.values,
                                                 add_special_tokens=True,
                                                 return_attention_mask=True,
                                                 pad_to_max_length=True,
                                                 max_length=256,
                                                 return_tensors='pt')
```

### 2. Model Initialization
Using `BertForSequenceClassification` with six output classes.
```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                       num_labels=len(label_dict),
                                                       output_attentions=False,
                                                       output_hidden_states=False)
```

### 3. Training Setup
Using AdamW optimizer and a learning rate scheduler.
```python
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
epochs = 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                             num_training_steps=len(dataloader_train) * epochs)
```

### 4. Training Loop
Training the model using PyTorch.
```python
for epoch in range(1, epochs+1):
    model.train()
    loss_train_total = 0
    
    for batch in dataloader_train:
        optimizer.zero_grad()
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
```

### 5. Evaluation
Calculating accuracy and F1-score.
```python
from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')
```

## Results and Conclusion
- Model is trained for **10 epochs**.
- Performance evaluated using **F1-score and accuracy**.
- The model can be fine-tuned further for better accuracy.

## References
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace BERT Documentation](https://huggingface.co/transformers/model_doc/bert.html)

## Usage
1. Install dependencies: `pip install torch transformers scikit-learn pandas`
2. Run the Jupyter notebook step by step.
3. Use trained model for sentiment analysis predictions.

