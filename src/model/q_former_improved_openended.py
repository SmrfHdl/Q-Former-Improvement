"""
Q-Former Improved for Open-Ended VQA.

This model extends the improved Q-Former (with SGG + NSM) for open-ended VQA.
Key changes: answer head outputs logits over answer vocabulary instead of binary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from layers.cross_modal_transformer import CrossModalTransformer
from model.clip_vit import VisionEncoder
from loguru import logger

# Import components from improved model
from model.q_former_improved import (
    GraphConvLayer,
    SpatialRelationEncoder,
    RelationTypePredictor,
    SceneGraphGenerator,
    ControlUnit,
    ReadUnit,
    WriteUnit,
    ReasoningCell,
    NeuralStateMachine,
    HierarchicalGate,
    ObjectDetectionPath,
)


class HierarchicalReasoningPathOpenEnded(nn.Module):
    """
    Level 3: Hierarchical Reasoning with Neural State Machine for Open-Ended VQA.
    
    Key difference: answer_head outputs over answer vocabulary instead of binary.
    """
    def __init__(self, dim: int, num_heads: int = 8, num_hops: int = 4,
                 num_global_queries: int = 32, vocab_size: int = 49408,
                 num_answers: int = 3129,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_answers = num_answers
        
        # Global queries for additional context
        self.global_queries = nn.Parameter(torch.randn(1, num_global_queries, dim))
        
        # Neural State Machine
        self.nsm = NeuralStateMachine(dim, num_heads, num_hops, dropout)
        
        # Cross-modal transformer for global context
        self.global_transformer = CrossModalTransformer(
            dim=dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout
        )
        
        # Feature fusion (NSM output + global features)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Task-specific heads
        # ITM head
        self.itm_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 2)
        )
        
        # LM head (for text generation)
        self.lm_head = nn.Linear(dim, vocab_size)
        
        # OPEN-ENDED ANSWER HEAD: Multi-class classification
        self.answer_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_answers)
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, object_features: torch.Tensor, relation_features: torch.Tensor,
                image_features: torch.Tensor, text_embeddings: torch.Tensor,
                attention_mask: torch.Tensor = None):
        """
        Args:
            object_features: From Level 1 (batch, num_obj, dim)
            relation_features: From Level 2 (batch, num_rel, dim)
            image_features: Original image features (batch, patches, dim)
            text_embeddings: Question embeddings (batch, seq_len, dim)
            attention_mask: Attention mask
            
        Returns:
            global_features, itm_logits, lm_logits, answer_logits, nsm_attention
        """
        batch = image_features.shape[0]
        
        # Create question mask from attention_mask
        question_mask = None
        
        # Neural State Machine reasoning
        nsm_output, memory_states, nsm_attention = self.nsm(
            text_embeddings, object_features, relation_features, question_mask
        )
        
        # Global context via transformer
        global_queries = self.global_queries.expand(batch, -1, -1).clone()
        
        # Combine hierarchical features for cross-attention
        hierarchical_features = torch.cat([object_features, relation_features], dim=1)
        
        global_features, updated_text = self.global_transformer(
            global_queries, hierarchical_features, text_embeddings, attention_mask
        )
        
        # Pool global features
        global_pooled = global_features.mean(dim=1)
        
        # Fuse NSM output with global features
        fused = self.fusion(torch.cat([nsm_output, global_pooled], dim=-1))
        fused = self.norm(fused)
        
        # Task predictions
        itm_logits = self.itm_head(fused)
        lm_logits = self.lm_head(updated_text)
        answer_logits = self.answer_head(fused)  # Now outputs (batch, num_answers)
        
        return global_features, itm_logits, lm_logits, answer_logits, nsm_attention


class QFormerImprovedOpenEnded(nn.Module):
    """
    Improved Q-Former for Open-Ended VQA with 3-Level Hierarchical Multi-Path Reasoning.
    
    Architecture:
        Level 1: Object Detection Path - Extracts object-level features
        Level 2: Scene Graph Generation - Builds relational graph with GNN
        Level 3: Neural State Machine - Multi-hop compositional reasoning
        
    Key difference from binary version:
        - Answer head outputs over vocabulary of answers instead of binary
        - Uses CrossEntropy loss instead of BCE
    """
    def __init__(
            self, 
            sequence_size: int,
            qformer_hidden_size: int,
            blocks_num: int,
            num_heads: int,
            num_answers: int,  # Size of answer vocabulary
            num_object_queries: int = 32,
            num_relation_queries: int = 64,
            num_global_queries: int = 32,
            num_reasoning_hops: int = 4,
            num_relation_types: int = 16,
            device: torch.device = torch.device('cuda'),
            use_clip_for_text: bool = True,
            clip_model_name: str = "openai/clip-vit-large-patch14",
            dropout_rate: float = 0.3,
            unfreeze_layers: int = 0,
            label_smoothing_answer: float = 0.1):
        super(QFormerImprovedOpenEnded, self).__init__()

        self.vision_dim = 1024
        self.device = device
        self.max_text_len = sequence_size
        self.use_clip_for_text = use_clip_for_text
        self.unfreeze_layers = unfreeze_layers
        self.qformer_hidden_size = qformer_hidden_size
        self.num_answers = num_answers
        self.label_smoothing_answer = label_smoothing_answer

        # Text encoder setup
        if self.use_clip_for_text:
            self._setup_clip_model(clip_model_name, unfreeze_layers)
        else:
            self._setup_bert_model(unfreeze_layers)

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=clip_model_name,
            device=device,
            unfreeze_layers=unfreeze_layers
        )
        
        # Projection layers
        self.vision_projection = nn.Linear(self.vision_dim, qformer_hidden_size)
        self.vision_norm = nn.LayerNorm(qformer_hidden_size)
        self.text_projection = nn.Linear(self.text_dim, qformer_hidden_size)
        self.text_norm = nn.LayerNorm(qformer_hidden_size)
        
        self.vision_dropout = nn.Dropout(dropout_rate * 0.5)
        self.text_dropout = nn.Dropout(dropout_rate * 0.5)

        # Distribute layers across levels
        layers_per_level = max(1, blocks_num // 3)
        
        # Level 1: Object Detection
        self.object_path = ObjectDetectionPath(
            dim=qformer_hidden_size,
            num_heads=num_heads,
            num_layers=layers_per_level,
            num_object_queries=num_object_queries,
            dropout=dropout_rate
        )
        
        # Level 2: Scene Graph Generation
        self.scene_graph = SceneGraphGenerator(
            dim=qformer_hidden_size,
            num_heads=num_heads,
            num_layers=layers_per_level,
            num_relation_types=num_relation_types,
            dropout=dropout_rate
        )
        
        # Level 3: Hierarchical Reasoning with NSM (Open-Ended version)
        self.reasoning_path = HierarchicalReasoningPathOpenEnded(
            dim=qformer_hidden_size,
            num_heads=num_heads,
            num_hops=num_reasoning_hops,
            num_global_queries=num_global_queries,
            vocab_size=self.tokenizer.vocab_size if hasattr(self, 'tokenizer') else 49408,
            num_answers=num_answers,
            dropout=dropout_rate
        )

        # ITC temperature
        self.register_buffer('temperature', torch.tensor(0.07))

        self.init_weights()
        self.to(device)
        
        self._log_model_info()

    def _log_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"QFormerImprovedOpenEnded initialized:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Answer vocabulary size: {self.num_answers}")
        logger.info(f"  - Level 1: Object Detection Path")
        logger.info(f"  - Level 2: Scene Graph Generation (GNN)")
        logger.info(f"  - Level 3: Neural State Machine (Open-Ended)")

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
        for i, block in enumerate(reversed(self.clip_model.text_model.encoder.layers)):
            if i < unfreeze_layers:
                for param in block.parameters():
                    param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze {unfreeze_layers} CLIP layers. Trainable params: {trainable_params}")

    def _unfreeze_bert_layers(self, unfreeze_layers: int):
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
    
    def generate_attention_mask(self, task: str, query_len: int, pad_mask: torch.Tensor, 
                                 device: torch.device = 'cpu'):
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
        Forward pass for open-ended VQA through 3-level hierarchical architecture.
        
        Args:
            samples: Dict with keys:
                - image_input: Dict with 'pixel_values'
                - question: List of question strings
                - answer_idx: Tensor of answer indices
        """
        image_input = samples['image_input']
        question = samples['question']

        # Encode vision and text
        image_features = self.vision_encoder.encode(image_input)
        batch_size = image_features.shape[0]
        image_features = self.vision_projection(image_features)
        image_features = self.vision_norm(image_features)
        image_features = self.vision_dropout(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-6)

        question_output, question_tokens = self.encode_text(question)
        text_embeddings = question_output['last_hidden_state']
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = self.text_norm(text_embeddings)
        text_embeddings = self.text_dropout(text_embeddings)

        # === LEVEL 1: Object Detection Path ===
        attention_mask_l1 = self.generate_attention_mask(
            task='itc',
            query_len=32,
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        object_features, spatial_info, object_confidence_logits = self.object_path(
            image_features, text_embeddings, attention_mask_l1
        )
        
        loss_object = F.binary_cross_entropy_with_logits(
            object_confidence_logits.squeeze(-1),
            torch.ones_like(object_confidence_logits.squeeze(-1)) * 0.5
        )
        
        object_confidence = torch.sigmoid(object_confidence_logits)

        # === LEVEL 2: Scene Graph Generation ===
        enriched_objects, edge_features, relation_logits = self.scene_graph(
            object_features, spatial_info, text_embeddings, 
            question_tokens['attention_mask']
        )
        
        num_obj = enriched_objects.shape[1]
        relation_features = edge_features.reshape(batch_size, num_obj * num_obj, -1)
        
        with torch.no_grad():
            relation_importance = relation_logits.max(dim=-1)[0]
            relation_importance = relation_importance.reshape(batch_size, -1)
            k = min(64, relation_importance.shape[1])
            _, top_indices = relation_importance.topk(k, dim=1)
        
        relation_features = torch.gather(
            relation_features, 1, 
            top_indices.unsqueeze(-1).expand(-1, -1, relation_features.shape[-1])
        )
        
        num_relation_types = relation_logits.shape[-1]
        uniform_target = torch.ones(batch_size, num_obj, num_obj, num_relation_types, 
                                    device=self.device) / num_relation_types
        loss_relation = F.kl_div(
            F.log_softmax(relation_logits, dim=-1),
            uniform_target,
            reduction='batchmean'
        )

        # === LEVEL 3: Neural State Machine Reasoning ===
        attention_mask_l3 = self.generate_attention_mask(
            task='itm',
            query_len=32,
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        global_features, itm_logits, lm_logits, answer_logits, nsm_attention = self.reasoning_path(
            enriched_objects, relation_features, 
            image_features, text_embeddings, attention_mask_l3
        )

        # === Image-Text Contrastive (ITC) Loss ===
        if self.use_clip_for_text:
            eos_token_id = 49407
            eos_positions = (question_tokens['input_ids'] == eos_token_id).int().argmax(dim=-1)
            batch_indices = torch.arange(batch_size, device=self.device)
            cls_text_embedding = question_output['last_hidden_state'][batch_indices, eos_positions, :]
        else:
            cls_text_embedding = question_output['last_hidden_state'][:, 0, :]
        cls_text_embedding = self.text_projection(cls_text_embedding)
        cls_text_embedding = self.text_norm(cls_text_embedding)
        cls_text_embedding = F.normalize(cls_text_embedding, p=2, dim=-1, eps=1e-6)
        
        global_image_embedding = global_features.mean(dim=1)
        global_image_embedding = F.normalize(global_image_embedding, p=2, dim=-1, eps=1e-6)
        
        sim_i2t = torch.matmul(global_image_embedding, cls_text_embedding.T)
        sim_i2t = torch.clamp(sim_i2t, min=-100, max=100) / self.temperature
        sim_t2i = sim_i2t.T
        
        targets = torch.arange(batch_size, device=self.device, dtype=torch.long)
        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
                    F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2

        # === Image-Text Matching (ITM) Loss ===
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

        neg_image_indices = []
        for b in range(batch_size):
            try:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            except RuntimeError:
                neg_idx = torch.argmax(weights_t2i[b]).item()
            neg_image_indices.append(neg_idx)
        
        neg_text_indices = []
        for b in range(batch_size):
            try:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            except RuntimeError:
                neg_idx = torch.argmax(weights_i2t[b]).item()
            neg_text_indices.append(neg_idx)
        
        neg_image_features = image_features[neg_image_indices]
        neg_text_embeddings = text_embeddings[torch.tensor(neg_text_indices, device=self.device)]
        
        _, itm_logits_neg1, _, _, _ = self.reasoning_path(
            enriched_objects, relation_features, neg_image_features, text_embeddings, attention_mask_l3
        )
        
        _, itm_logits_neg2, _, _, _ = self.reasoning_path(
            enriched_objects, relation_features, image_features, neg_text_embeddings, attention_mask_l3
        )
        
        itm_logits_all = torch.cat([itm_logits, itm_logits_neg1, itm_logits_neg2], dim=0)
        itm_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long, device=self.device),
            torch.zeros(batch_size, dtype=torch.long, device=self.device),
            torch.zeros(batch_size, dtype=torch.long, device=self.device)
        ], dim=0)
        
        itm_class_weights = torch.tensor([1.0, 2.0], device=self.device)
        loss_itm = F.cross_entropy(itm_logits_all, itm_labels, weight=itm_class_weights)
        
        itm_predictions = torch.argmax(itm_logits_all, dim=-1)
        itm_accuracy = (itm_predictions == itm_labels).float().mean()

        # === Image Grounded Text Generation (IGT) Loss ===
        igt_input_ids = question_tokens['input_ids'].clone()
        igt_input_ids[:, 0] = self.dec_token_id
        labels = igt_input_ids.clone()
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        
        shifted_logits = lm_logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        loss_igt = F.cross_entropy(
            shifted_logits.reshape(-1, lm_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100
        )

        # === OPEN-ENDED ANSWER PREDICTION ===
        answer_labels = samples['answer_idx'].to(self.device)
        
        loss_answer = F.cross_entropy(
            answer_logits,
            answer_labels,
            label_smoothing=self.label_smoothing_answer
        )
        
        answer_predictions = torch.argmax(answer_logits, dim=-1)
        answer_accuracy = (answer_predictions == answer_labels).float().mean()

        # === Total Loss ===
        total_loss = (0.05 * loss_object +
                      0.1 * loss_relation +
                      1.0 * loss_itc +
                      1.0 * loss_itm +
                      0.0 * loss_igt +
                      2.0 * loss_answer)

        return {
            'answer_accuracy': answer_accuracy,
            'itm_accuracy': itm_accuracy,
            'loss_answer': loss_answer,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'loss_igt': loss_igt,
            'loss_object': loss_object,
            'loss_relation': loss_relation,
            'total_loss': total_loss,
            'answer_predictions': answer_predictions.detach(),
            'answer_logits': answer_logits.detach(),
            'answer_labels': answer_labels.detach(),
            'object_confidence': object_confidence.detach(),
            'spatial_info': spatial_info.detach(),
            'relation_logits': relation_logits.detach(),
            'nsm_attention': nsm_attention,
        }

