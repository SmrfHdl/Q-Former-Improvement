"""
Q-Former Base for Open-Ended VQA.

This model treats VQA as a classification task over a vocabulary of answers.
Standard approach: classify into top-K most frequent answers (typically K=3129 for VQA v2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from layers.cross_modal_transformer import CrossModalTransformer
from model.clip_vit import VisionEncoder
from loguru import logger


class QFormerBaseOpenEnded(nn.Module):
    """
    Q-Former Base for Open-Ended VQA with answer classification.
    
    Key differences from binary VQA:
    - Answer head outputs logits over answer vocabulary (not binary)
    - Uses CrossEntropy loss instead of BCE
    - Supports variable answer vocabulary sizes
    """
    def __init__(
            self, 
            sequence_size: int,
            qformer_hidden_size: int,
            blocks_num: int,
            num_heads: int,
            num_queries: int,
            num_answers: int,  # Size of answer vocabulary
            device: torch.device,
            use_clip_for_text: bool = True,
            clip_model_name: str = "openai/clip-vit-large-patch14",
            dropout_rate: float = 0.3,
            unfreeze_layers: int = 0,
            learnable_temperature: bool = True,
            initial_temperature: float = 0.07,
            temperature_min: float = 0.01,
            temperature_max: float = 0.5,
            label_smoothing_answer: float = 0.1,
            label_smoothing_itc: float = 0.1,
            label_smoothing_itm: float = 0.1,
            loss_weight_itc: float = 0.2,
            loss_weight_itm: float = 0.3,
            loss_weight_igt: float = 0.0,
            loss_weight_answer: float = 1.0,
            stochastic_depth_rate: float = 0.1):
        super(QFormerBaseOpenEnded, self).__init__()

        self.vision_dim = 1024  # ViT default
        self.device = device
        self.max_text_len = sequence_size
        self.use_clip_for_text = use_clip_for_text
        self.unfreeze_layers = unfreeze_layers
        self.num_answers = num_answers
        
        # Store loss weights and label smoothing
        self.label_smoothing_answer = label_smoothing_answer
        self.label_smoothing_itc = label_smoothing_itc
        self.label_smoothing_itm = label_smoothing_itm
        self.loss_weight_itc = loss_weight_itc
        self.loss_weight_itm = loss_weight_itm
        self.loss_weight_igt = loss_weight_igt
        self.loss_weight_answer = loss_weight_answer
        
        # Temperature bounds for clamping
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

        if self.use_clip_for_text:
            self._setup_clip_model(clip_model_name, unfreeze_layers)
        else:
            self._setup_bert_model(unfreeze_layers)

        self.vision_encoder = VisionEncoder(
            model_name=clip_model_name,
            device=device,
            unfreeze_layers=unfreeze_layers
        )
        
        self.vision_projection = nn.Linear(self.vision_dim, qformer_hidden_size)
        self.vision_norm = nn.LayerNorm(qformer_hidden_size)
        self.text_projection = nn.Linear(self.text_dim, qformer_hidden_size)
        self.text_norm = nn.LayerNorm(qformer_hidden_size)
        
        self.vision_dropout = nn.Dropout(dropout_rate)
        self.text_dropout = nn.Dropout(dropout_rate)
        self.feature_dropout = nn.Dropout(dropout_rate * 0.5)

        self.learned_queries = nn.Parameter(
            torch.randn(1, num_queries, qformer_hidden_size)
        )

        self.cross_modal_transformer = CrossModalTransformer(
            qformer_hidden_size,
            num_heads,
            blocks_num,
            dropout=dropout_rate,
            stochastic_depth_rate=stochastic_depth_rate
        )

        # Learnable temperature
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature)))
            logger.info(f"Using LEARNABLE temperature, initial value: {initial_temperature}")
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(initial_temperature)))
            logger.info(f"Using FIXED temperature: {initial_temperature}")
        
        self.learnable_temperature = learnable_temperature

        # ITM head
        self.itm_head = nn.Sequential(
            nn.Linear(qformer_hidden_size, qformer_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(qformer_hidden_size // 2, 2)
        )

        # LM head
        self.lm_head = nn.Linear(qformer_hidden_size, self.tokenizer.vocab_size)

        # OPEN-ENDED ANSWER HEAD: Multi-class classification over answer vocabulary
        self.answer_head = nn.Sequential(
            nn.Linear(qformer_hidden_size, qformer_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(qformer_hidden_size, qformer_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(qformer_hidden_size // 2, num_answers)
        )

        self.init_weights()
        self.to(device)
        
        logger.info(f"QFormerBaseOpenEnded initialized with {num_answers} answer classes")

    def _setup_clip_model(self, clip_model_name: str, unfreeze_clip_layers: int):
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_tokenizer = self.clip_processor.tokenizer
        self.clip_tokenizer.add_special_tokens({'additional_special_tokens': ["[DEC]"]})

        new_vocab_size = len(self.clip_tokenizer)
        old_vocab_size = self.clip_model.text_model.embeddings.token_embedding.weight.shape[0]

        if new_vocab_size != old_vocab_size:
            old_embeddings = self.clip_model.text_model.embeddings.token_embedding
            embedding_dim = old_embeddings.embedding_dim
            new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
            new_embeddings.weight.data[:old_vocab_size] = old_embeddings.weight.data
            self.clip_model.text_model.embeddings.token_embedding = new_embeddings

        self.dec_token_id = self.clip_tokenizer.convert_tokens_to_ids('[DEC]')

        for param in self.clip_model.parameters():
            param.requires_grad = False

        if unfreeze_clip_layers > 0:
            self._unfreeze_clip_layers(unfreeze_clip_layers)

        self.text_dim = self.clip_model.text_model.config.hidden_size
        self.tokenizer = self.clip_tokenizer

    def _setup_bert_model(self, unfreeze_bert_layers: int):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.dec_token_id = self.tokenizer.convert_tokens_to_ids('[DEC]')

        for param in self.bert.parameters():
            param.requires_grad = False

        if unfreeze_bert_layers > 0:
            self._unfreeze_bert_layers(unfreeze_bert_layers)

        self.text_dim = self.bert.config.hidden_size

    def _unfreeze_clip_layers(self, unfreeze_layers: int):
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for i, block in enumerate(reversed(self.clip_model.text_model.encoder.layers)):
            if i < unfreeze_layers:
                for param in block.parameters():
                    param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze {unfreeze_layers} CLIP layers. Trainable params: {trainable_params}")

    def _unfreeze_bert_layers(self, unfreeze_layers: int):
        for param in self.bert.parameters():
            param.requires_grad = False
        for i, layer in enumerate(reversed(self.bert.encoder.layer)):
            if i < unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze {unfreeze_layers} BERT layers. Trainable params: {trainable_params}")

    def init_weights(self):
        nn.init.xavier_uniform_(self.vision_projection.weight)
        nn.init.zeros_(self.vision_projection.bias)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)
        nn.init.normal_(self.learned_queries, std=0.02)

        for layer in self.itm_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

        for layer in self.answer_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def encode_text(self, questions: list[str] | str):
        if self.use_clip_for_text:
            question_tokens = self.clip_processor(
                text=questions,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            question_tokens = {k: v.to(self.device) for k, v in question_tokens.items()}
            question_output = self.clip_model.text_model(
                input_ids=question_tokens['input_ids'],
                attention_mask=question_tokens['attention_mask'],
                output_hidden_states=True
            )
        else:
            question_tokens = self.tokenizer(
                questions,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            question_tokens = {k: v.to(self.device) for k, v in question_tokens.items()}
            question_output = self.bert(
                input_ids=question_tokens['input_ids'],
                attention_mask=question_tokens['attention_mask'],
                return_dict=True
            )
        return question_output, question_tokens
    
    def generate_attention_mask(self, task: str, query_len: int, pad_mask: torch.Tensor, device: torch.device = 'cpu'):
        batch_size, text_len = pad_mask.size()
        total_len = query_len + text_len 
        MASK_VALUE = -1e9
        task_mask = torch.zeros((batch_size, total_len, total_len), device=device)

        if task == 'itm':
            pass
        elif task == "igt":
            causal_indices = torch.triu_indices(text_len, text_len, offset=1, device=device)
            for b in range(batch_size):
                task_mask[b, query_len + causal_indices[0], query_len + causal_indices[1]] = MASK_VALUE
            task_mask[:, :query_len, query_len:] = MASK_VALUE
        elif task == 'itc':
            task_mask[:, :query_len, query_len:] = MASK_VALUE
            task_mask[:, query_len:, :query_len] = MASK_VALUE

        padding_positions = (pad_mask == 0)
        for b in range(batch_size):
            if padding_positions[b].any():
                pad_indices = torch.nonzero(padding_positions[b], as_tuple=True)[0]
                task_mask[b, :, query_len + pad_indices] = MASK_VALUE
                task_mask[b, query_len + pad_indices, :] = MASK_VALUE

        return task_mask
    
    def forward(self, samples: dict):
        """
        Forward pass for open-ended VQA.
        
        Args:
            samples: Dict with keys:
                - image_input: Dict with 'pixel_values'
                - question: List of question strings
                - answer_idx: Tensor of answer indices (for training)
        """
        image_input = samples['image_input']
        image_features = self.vision_encoder.encode(image_input)
        
        question = samples['question']
        batch_size = image_features.shape[0]

        # Project and normalize vision features
        image_features = self.vision_projection(image_features)
        image_features = self.vision_norm(image_features)
        image_features = self.vision_dropout(image_features)

        queries = self.learned_queries.expand(batch_size, -1, -1).clone()

        question_output, question_tokens = self.encode_text(question)
        
        # Project text embeddings
        text_embeddings_raw = question_output['last_hidden_state']
        text_embeddings_projected = self.text_projection(text_embeddings_raw)
        text_embeddings_projected = self.text_norm(text_embeddings_projected)
        text_embeddings_projected = self.text_dropout(text_embeddings_projected)

        # Get CLS embedding for ITC
        if self.use_clip_for_text:
            eos_token_id = 49407
            eos_positions = (question_tokens['input_ids'] == eos_token_id).int().argmax(dim=-1)
            batch_indices = torch.arange(text_embeddings_projected.size(0), device=self.device)
            cls_text_embedding = text_embeddings_projected[batch_indices, eos_positions, :]
        else:
            cls_text_embedding = text_embeddings_projected[:, 0, :]
        
        cls_text_embedding_normalized = F.normalize(cls_text_embedding, p=2, dim=-1, eps=1e-6)

        attention_mask = self.generate_attention_mask(
            task='itc',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        queries, _ = self.cross_modal_transformer(
            queries,
            image_features,
            text_embeddings=text_embeddings_projected,
            attention_mask=attention_mask
        )

        # Keep un-normalized queries for answer prediction
        queries_for_answer = queries
        
        # L2 normalize for ITC
        queries_normalized = F.normalize(queries, p=2, dim=-1, eps=1e-6)

        # ITC similarity
        sim_i2t = torch.einsum("bqd, Bd -> bBq", queries_normalized, cls_text_embedding_normalized)
        sim_i2t, _ = sim_i2t.max(-1)
        
        temperature = torch.exp(self.log_temperature)
        temperature = torch.clamp(temperature, min=self.temperature_min, max=self.temperature_max)
        
        sim_i2t = torch.clamp(sim_i2t, min=-100, max=100)
        sim_i2t = sim_i2t / temperature
        sim_t2i = sim_i2t.T

        targets = torch.arange(batch_size, device=image_features.device, dtype=int)

        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=self.label_smoothing_itc) +
                    F.cross_entropy(sim_t2i, targets, label_smoothing=self.label_smoothing_itc)) / 2
        
        # ITM with hard negatives
        with torch.no_grad():
            sim_i2t_clone = sim_i2t.clone()
            sim_t2i_clone = sim_t2i.clone()
            sim_i2t_clone.fill_diagonal_(-10000)
            sim_t2i_clone.fill_diagonal_(-10000)
            sim_i2t_clone = torch.clamp(sim_i2t_clone, min=-100, max=100)
            sim_t2i_clone = torch.clamp(sim_t2i_clone, min=-100, max=100)

        weights_t2i = torch.softmax(sim_t2i_clone, dim=-1)
        weights_i2t = torch.softmax(sim_i2t_clone, dim=-1)
        
        weights_t2i = torch.where(torch.isnan(weights_t2i) | torch.isinf(weights_t2i), 
                                  torch.ones_like(weights_t2i) / batch_size, weights_t2i)
        weights_i2t = torch.where(torch.isnan(weights_i2t) | torch.isinf(weights_i2t), 
                                  torch.ones_like(weights_i2t) / batch_size, weights_i2t)
        
        weights_t2i = weights_t2i + 1e-8
        weights_i2t = weights_i2t + 1e-8
        weights_t2i = weights_t2i / weights_t2i.sum(dim=-1, keepdim=True)
        weights_i2t = weights_i2t / weights_i2t.sum(dim=-1, keepdim=True)

        # Sample hard negatives
        image_embeddings_negative = []
        for b in range(batch_size):
            try:
                negative_idx = torch.multinomial(weights_t2i[b], 1).item()
            except RuntimeError:
                negative_idx = torch.argmax(weights_t2i[b]).item()
            image_embeddings_negative.append(image_features[negative_idx])
        image_embeddings_negative = torch.stack(image_embeddings_negative, dim=0)

        text_embeddings_negative = []
        attention_masks_negative = []
        for b in range(batch_size):
            try:
                negative_idx = torch.multinomial(weights_i2t[b], 1).item()
            except RuntimeError:
                negative_idx = torch.argmax(weights_i2t[b]).item()
            text_embeddings_negative.append(text_embeddings_projected[negative_idx])
            attention_masks_negative.append(question_tokens['attention_mask'][negative_idx])
        text_embeddings_negative = torch.stack(text_embeddings_negative, dim=0)
        attention_masks_negative = torch.stack(attention_masks_negative, dim=0)
        attention_masks_negative = self.generate_attention_mask(
            task='itm',
            query_len=queries.shape[1],
            pad_mask=attention_masks_negative,
            device=self.device
        )

        # ITM forward passes
        attention_mask_neg1 = self.generate_attention_mask(
            task='itm',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        queries_neg1 = self.learned_queries.expand(batch_size, -1, -1).clone()
        queries_neg1, _ = self.cross_modal_transformer(
            queries_neg1,
            image_embeddings_negative,
            text_embeddings=text_embeddings_projected,
            attention_mask=attention_mask_neg1
        )
        
        queries_neg2 = self.learned_queries.expand(batch_size, -1, -1).clone()
        queries_neg2, _ = self.cross_modal_transformer(
            queries_neg2,
            image_features,
            text_embeddings=text_embeddings_negative,
            attention_mask=attention_masks_negative
        )
        
        # ITM predictions
        queries_pos_cls = queries[:, 0, :]
        queries_neg1_cls = queries_neg1[:, 0, :]
        queries_neg2_cls = queries_neg2[:, 0, :]
        
        itm_cls_features = torch.cat([queries_pos_cls, queries_neg1_cls, queries_neg2_cls], dim=0)
        logits = self.itm_head(itm_cls_features)

        itm_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long)
        ], dim=0).to(self.device)

        itm_class_weights = torch.tensor([1.0, 2.0], device=self.device)
        loss_itm = F.cross_entropy(logits, itm_labels, weight=itm_class_weights,
                                   label_smoothing=self.label_smoothing_itm)
        
        itm_predictions = torch.argmax(logits, dim=-1)
        itm_accuracy = (itm_predictions == itm_labels).float().mean()

        # IGT Loss
        igt_input_ids = question_tokens['input_ids'].clone()
        igt_input_ids[:, 0] = self.dec_token_id
        labels = igt_input_ids.clone()
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        if self.use_clip_for_text:
            igt_text_output = self.clip_model.text_model(
                input_ids=igt_input_ids,
                attention_mask=question_tokens['attention_mask']
            )
        else:
            igt_text_output = self.bert(
                input_ids=igt_input_ids,
                attention_mask=question_tokens['attention_mask'],
                return_dict=True
            )
        
        igt_text_projected = self.text_projection(igt_text_output['last_hidden_state'])
        igt_text_projected = self.text_norm(igt_text_projected)
        igt_text_projected = self.text_dropout(igt_text_projected)
        
        igt_attention_mask = self.generate_attention_mask(
            task='igt',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )

        queries_igt = self.learned_queries.expand(batch_size, -1, -1).clone()
        queries_igt, text_embeddings_igt = self.cross_modal_transformer(
            queries_igt,
            image_features,
            text_embeddings=igt_text_projected,
            attention_mask=igt_attention_mask
        )

        text_logits = self.lm_head(text_embeddings_igt)
        shifted_logits = text_logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        loss_igt = F.cross_entropy(
            shifted_logits.reshape(-1, self.tokenizer.vocab_size),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction='mean'
        )

        # === OPEN-ENDED ANSWER PREDICTION ===
        # Max pool over queries for answer prediction
        max_pooled_queries = torch.max(queries_for_answer, dim=1)[0]  # (batch_size, dim)
        
        # Get answer logits over vocabulary
        answer_logits = self.answer_head(max_pooled_queries)  # (batch_size, num_answers)

        # Get answer labels from samples
        answer_labels = samples['answer_idx'].to(self.device)  # (batch_size,)
        
        # Cross entropy loss with label smoothing
        loss_answer = F.cross_entropy(
            answer_logits,
            answer_labels,
            label_smoothing=self.label_smoothing_answer
        )

        # Calculate accuracy
        answer_predictions = torch.argmax(answer_logits, dim=-1)
        answer_accuracy = (answer_predictions == answer_labels).float().mean()

        # Total loss
        total_loss = (self.loss_weight_itc * loss_itc + 
                      self.loss_weight_itm * loss_itm + 
                      self.loss_weight_igt * loss_igt + 
                      self.loss_weight_answer * loss_answer)
        
        # Get current temperature for logging
        temperature = torch.exp(self.log_temperature)
        temperature = torch.clamp(temperature, min=self.temperature_min, max=self.temperature_max)

        return {
            'answer_accuracy': answer_accuracy,
            'itm_accuracy': itm_accuracy,
            'loss_answer': loss_answer,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'loss_igt': loss_igt,
            'total_loss': total_loss,
            'answer_predictions': answer_predictions.detach(),
            'answer_logits': answer_logits.detach(),
            'answer_labels': answer_labels.detach(),
            'temperature': temperature.detach(),
        }

