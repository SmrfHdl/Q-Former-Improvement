import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from layers.cross_modal_transformer import CrossModalTransformer
from model.clip_vit import VisionEncoder
from loguru import logger


class HierarchicalGate(nn.Module):
    """
    Gating mechanism to control information flow between hierarchical levels.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, lower_level: torch.Tensor, current_level: torch.Tensor):
        """
        Args:
            lower_level: Features from lower hierarchy (batch, seq, dim)
            current_level: Features from current level (batch, seq, dim)
        Returns:
            Gated combination of both levels
        """
        combined = torch.cat([lower_level, current_level], dim=-1)
        gate_values = self.gate(combined)
        return gate_values * lower_level + (1 - gate_values) * current_level


class ObjectDetectionPath(nn.Module):
    """
    Level 1: Object Detection Path
    Extracts object-level features with spatial and attribute information.
    """
    def __init__(self, dim: int, num_heads: int, num_layers: int, num_object_queries: int, dropout: float = 0.1):
        super().__init__()
        
        self.object_queries = nn.Parameter(torch.randn(1, num_object_queries, dim))
        
        # Cross-modal transformer for object detection
        self.object_transformer = CrossModalTransformer(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Object attribute heads
        self.spatial_head = nn.Linear(dim, 4)  # x, y, w, h
        self.confidence_head = nn.Linear(dim, 1)  # objectness score
        
    def forward(self, image_features: torch.Tensor, text_embeddings: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            image_features: Visual features (batch, patches, dim)
            text_embeddings: Text embeddings (batch, seq_len, dim)
            attention_mask: Attention mask
        Returns:
            object_features: Object-level features (batch, num_objects, dim)
            spatial_info: Spatial bounding boxes (batch, num_objects, 4)
            confidence: Objectness scores (batch, num_objects, 1)
        """
        batch_size = image_features.shape[0]
        object_queries = self.object_queries.expand(batch_size, -1, -1).clone()
        
        # Extract object features through cross-attention with image
        object_features, _ = self.object_transformer(
            object_queries,
            image_features,
            text_embeddings,
            attention_mask
        )
        
        # Predict spatial information and confidence
        spatial_info = torch.sigmoid(self.spatial_head(object_features))  # Normalize to [0, 1]
        confidence = torch.sigmoid(self.confidence_head(object_features))
        
        return object_features, spatial_info, confidence


class RelationReasoningPath(nn.Module):
    """
    Level 2: Relation Reasoning Path
    Models pairwise relations between objects and regional semantics.
    """
    def __init__(self, dim: int, num_heads: int, num_layers: int, num_relation_queries: int, dropout: float = 0.1):
        super().__init__()
        
        self.relation_queries = nn.Parameter(torch.randn(1, num_relation_queries, dim))
        
        # Pairwise relation encoder
        self.pairwise_encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Spatial relation encoder
        self.spatial_relation_encoder = nn.Sequential(
            nn.Linear(8, dim // 4),  # 4 (bbox1) + 4 (bbox2)
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4)
        )
        
        # Relation transformer
        self.relation_transformer = CrossModalTransformer(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Relation type classifier (spatial, semantic, functional)
        self.relation_classifier = nn.Linear(dim, 3)
        
        # Gate for integrating object features
        self.object_gate = HierarchicalGate(dim)
        
    def forward(self, object_features: torch.Tensor, spatial_info: torch.Tensor, 
                image_features: torch.Tensor, text_embeddings: torch.Tensor, 
                attention_mask: torch.Tensor = None):
        """
        Args:
            object_features: Object features from Level 1 (batch, num_objects, dim)
            spatial_info: Spatial bounding boxes (batch, num_objects, 4)
            image_features: Image features (batch, patches, dim)
            text_embeddings: Text embeddings (batch, seq_len, dim)
            attention_mask: Attention mask
        Returns:
            relation_features: Relation-level features (batch, num_relations, dim)
            relation_types: Predicted relation types (batch, num_relations, 3)
        """
        batch_size = object_features.shape[0]
        num_objects = object_features.shape[1]
        
        # Compute pairwise object relations
        # Create all pairs: (obj_i, obj_j)
        obj_i = object_features.unsqueeze(2).expand(-1, -1, num_objects, -1)  # (B, N, N, D)
        obj_j = object_features.unsqueeze(1).expand(-1, num_objects, -1, -1)  # (B, N, N, D)
        pairwise_features = torch.cat([obj_i, obj_j], dim=-1)  # (B, N, N, 2D)
        
        # Encode pairwise semantic relations
        pairwise_relations = self.pairwise_encoder(pairwise_features)  # (B, N, N, D)
        
        # Encode spatial relations
        spatial_i = spatial_info.unsqueeze(2).expand(-1, -1, num_objects, -1)  # (B, N, N, 4)
        spatial_j = spatial_info.unsqueeze(1).expand(-1, num_objects, -1, -1)  # (B, N, N, 4)
        spatial_pairs = torch.cat([spatial_i, spatial_j], dim=-1)  # (B, N, N, 8)
        spatial_relations = self.spatial_relation_encoder(spatial_pairs)  # (B, N, N, D//4)
        
        # Combine semantic and spatial relations
        # Pad spatial relations to match dimension
        spatial_relations_padded = F.pad(spatial_relations, (0, pairwise_relations.shape[-1] - spatial_relations.shape[-1]))
        combined_relations = pairwise_relations + spatial_relations_padded
        
        # Flatten pairwise relations to sequence
        relation_sequence = combined_relations.reshape(batch_size, num_objects * num_objects, -1)
        
        # Initialize relation queries
        relation_queries = self.relation_queries.expand(batch_size, -1, -1).clone()
        
        # Apply gating with object features
        # Use mean pooling of object features as context
        object_context = object_features.mean(dim=1, keepdim=True).expand(-1, relation_queries.shape[1], -1)
        relation_queries = self.object_gate(object_context, relation_queries)
        
        # Process through relation transformer
        relation_features, _ = self.relation_transformer(
            relation_queries,
            relation_sequence,
            text_embeddings,
            attention_mask
        )
        
        # Classify relation types
        relation_types = self.relation_classifier(relation_features)
        
        return relation_features, relation_types


class GlobalReasoningPath(nn.Module):
    """
    Level 3: Global Reasoning Path
    Performs holistic scene understanding and multi-hop reasoning for VQA.
    """
    def __init__(self, dim: int, num_heads: int, num_layers: int, num_global_queries: int, 
                 vocab_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.global_queries = nn.Parameter(torch.randn(1, num_global_queries, dim))
        
        # Global reasoning transformer
        self.global_transformer = CrossModalTransformer(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Multi-hop reasoning with self-attention
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)  # 2 hops for reasoning
        ])
        
        # Gates for hierarchical integration
        self.object_gate = HierarchicalGate(dim)
        self.relation_gate = HierarchicalGate(dim)
        
        # Task-specific heads
        self.itm_head = nn.Linear(dim, 2)  # Image-Text Matching
        self.lm_head = nn.Linear(dim, vocab_size)  # Language Modeling
        self.answer_head = nn.Linear(dim, 1)  # Binary Answer Prediction
        
    def forward(self, object_features: torch.Tensor, relation_features: torch.Tensor,
                image_features: torch.Tensor, text_embeddings: torch.Tensor,
                attention_mask: torch.Tensor = None):
        """
        Args:
            object_features: Features from Level 1 (batch, num_objects, dim)
            relation_features: Features from Level 2 (batch, num_relations, dim)
            image_features: Image features (batch, patches, dim)
            text_embeddings: Text embeddings (batch, seq_len, dim)
            attention_mask: Attention mask
        Returns:
            global_features: Global reasoning features (batch, num_global, dim)
            itm_logits: Image-text matching logits
            lm_logits: Language modeling logits
            answer_logits: Answer prediction logits
        """
        batch_size = image_features.shape[0]
        
        # Initialize global queries
        global_queries = self.global_queries.expand(batch_size, -1, -1).clone()
        
        # Hierarchical gating: integrate object features
        object_context = object_features.mean(dim=1, keepdim=True).expand(-1, global_queries.shape[1], -1)
        global_queries = self.object_gate(object_context, global_queries)
        
        # Hierarchical gating: integrate relation features
        relation_context = relation_features.mean(dim=1, keepdim=True).expand(-1, global_queries.shape[1], -1)
        global_queries = self.relation_gate(relation_context, global_queries)
        
        # Combine all hierarchical features
        hierarchical_features = torch.cat([object_features, relation_features], dim=1)
        
        # Global reasoning with cross-modal transformer
        global_features, updated_text = self.global_transformer(
            global_queries,
            hierarchical_features,
            text_embeddings,
            attention_mask
        )
        
        # Multi-hop reasoning
        for reasoning_layer in self.reasoning_layers:
            global_features = reasoning_layer(global_features)
        
        # Task-specific predictions
        # ITM: Use mean pooled global features
        itm_logits = self.itm_head(global_features.mean(dim=1))
        
        # LM: Use updated text embeddings
        lm_logits = self.lm_head(updated_text)
        
        # Answer: Use max pooled global features
        answer_logits = self.answer_head(global_features.max(dim=1)[0])
        
        return global_features, itm_logits, lm_logits, answer_logits


class QFormerImproved(nn.Module):
    """
    Improved Q-Former with 3-Level Hierarchical Multi-Path Reasoning.
    
    Architecture:
        Level 1: Object Detection Path - Extracts object-level features
        Level 2: Relation Reasoning Path - Models object relations
        Level 3: Global Reasoning Path - Holistic understanding and VQA
    """
    def __init__(
            self, 
            sequence_size: int,
            qformer_hidden_size: int,
            blocks_num: int,
            num_heads: int,
            num_object_queries: int = 32,
            num_relation_queries: int = 64,
            num_global_queries: int = 32,
            device: torch.device = torch.device('cuda'),
            use_clip_for_text: bool = True,
            clip_model_name: str = "openai/clip-vit-large-patch14",
            dropout_rate: float = 0.3,
            unfreeze_layers: int = 0):
        super(QFormerImproved, self).__init__()

        self.vision_dim = 1024  # ViT default
        self.device = device
        self.max_text_len = sequence_size
        self.use_clip_for_text = use_clip_for_text
        self.unfreeze_layers = unfreeze_layers
        self.qformer_hidden_size = qformer_hidden_size

        # Text encoder setup (must be done first to set self.text_dim)
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
        
        # Projection layers (after text encoder setup so self.text_dim is available)
        self.vision_projection = nn.Linear(self.vision_dim, qformer_hidden_size)
        self.vision_norm = nn.LayerNorm(qformer_hidden_size)
        self.text_projection = nn.Linear(self.text_dim, qformer_hidden_size)
        self.text_norm = nn.LayerNorm(qformer_hidden_size)
        
        self.vision_dropout = nn.Dropout(dropout_rate * 0.5)
        self.text_dropout = nn.Dropout(dropout_rate * 0.5)

        # 3-Level Hierarchical Paths
        layers_per_level = blocks_num // 3  # Distribute layers across 3 levels
        
        # Level 1: Object Detection
        self.object_path = ObjectDetectionPath(
            dim=qformer_hidden_size,
            num_heads=num_heads,
            num_layers=layers_per_level,
            num_object_queries=num_object_queries,
            dropout=dropout_rate
        )
        
        # Level 2: Relation Reasoning
        self.relation_path = RelationReasoningPath(
            dim=qformer_hidden_size,
            num_heads=num_heads,
            num_layers=layers_per_level,
            num_relation_queries=num_relation_queries,
            dropout=dropout_rate
        )
        
        # Level 3: Global Reasoning
        self.global_path = GlobalReasoningPath(
            dim=qformer_hidden_size,
            num_heads=num_heads,
            num_layers=layers_per_level,
            num_global_queries=num_global_queries,
            vocab_size=self.tokenizer.vocab_size if hasattr(self, 'tokenizer') else 49408,
            dropout=dropout_rate
        )

        # ITC temperature (fixed buffer)
        self.register_buffer('temperature', torch.tensor(0.07))

        self.init_weights()
        self.to(device)

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
        # Use Xavier uniform initialization for projection layers (standard for linear projections)
        # No aggressive scaling - let LayerNorm handle normalization
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
    
    def generate_attention_mask(self, task: str, query_len: int, pad_mask: torch.Tensor, device: torch.device = 'cpu'):
        """Generate attention mask based on task type."""
        batch_size, text_len = pad_mask.size()
        total_len = query_len + text_len 
        # Use a large finite negative value instead of -inf to avoid NaN in softmax backward
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
        Forward pass through 3-level hierarchical architecture.
        
        Level 1: Object Detection
        Level 2: Relation Reasoning  
        Level 3: Global Reasoning & VQA
        """
        image_input = samples['image_input']
        question = samples['question']
        batch_size = image_input.shape[0]

        # Encode vision and text
        image_features = self.vision_encoder.encode(image_input)
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
            query_len=32,  # num_object_queries
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        object_features, spatial_info, object_confidence = self.object_path(
            image_features, text_embeddings, attention_mask_l1
        )
        
        # Auxiliary loss for object detection (objectness)
        loss_object = F.binary_cross_entropy(
            object_confidence.squeeze(-1),
            torch.ones_like(object_confidence.squeeze(-1)) * 0.5  # Weak supervision
        )

        # === LEVEL 2: Relation Reasoning Path ===
        attention_mask_l2 = self.generate_attention_mask(
            task='itc',
            query_len=64,  # num_relation_queries
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        relation_features, relation_types = self.relation_path(
            object_features, spatial_info, image_features, text_embeddings, attention_mask_l2
        )
        
        # Auxiliary loss for relation classification (uniform distribution as weak supervision)
        loss_relation = F.cross_entropy(
            relation_types.reshape(-1, 3),
            torch.full((relation_types.shape[0] * relation_types.shape[1],), 
                      1, dtype=torch.long, device=self.device)  # Target middle class
        )

        # === LEVEL 3: Global Reasoning Path ===
        attention_mask_l3 = self.generate_attention_mask(
            task='itm',
            query_len=32,  # num_global_queries
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        global_features, itm_logits, lm_logits, answer_logits = self.global_path(
            object_features, relation_features, image_features, text_embeddings, attention_mask_l3
        )

        # === Image-Text Contrastive (ITC) Loss ===
        # Project text to same space as Q-Former features
        # FIX: Use EOS token position for CLIP (not position 0!)
        if self.use_clip_for_text:
            eos_token_id = 49407
            eos_positions = (question_tokens['input_ids'] == eos_token_id).int().argmax(dim=-1)
            batch_indices = torch.arange(batch_size, device=self.device)
            cls_text_embedding = question_output['last_hidden_state'][batch_indices, eos_positions, :]
        else:
            cls_text_embedding = question_output['last_hidden_state'][:, 0, :]
        cls_text_embedding = self.text_projection(cls_text_embedding)  # FIX: Project to Q-Former space
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

        # Sample hard negatives and compute ITM for positive + negative pairs
        itm_labels_pos = torch.ones(batch_size, dtype=torch.long, device=self.device)
        loss_itm = F.cross_entropy(itm_logits, itm_labels_pos)

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

        # === Answer Prediction Loss ===
        answers = samples['answer']
        answer_dict = {'yes': 1, 'no': 0}
        answer_labels = torch.tensor(
            [answer_dict[ans] for ans in answers],
            dtype=torch.float, device=self.device
        ).unsqueeze(1)
        
        loss_answer = F.binary_cross_entropy_with_logits(answer_logits, answer_labels)
        
        predictions = (torch.sigmoid(answer_logits) > 0.5).float()
        answer_accuracy = (predictions == answer_labels).float().mean()

        # === Total Loss with Hierarchical Weighting ===
        # Level 1 auxiliary: 0.1
        # Level 2 auxiliary: 0.1
        # Level 3 main tasks: higher weights
        total_loss = (0.1 * loss_object +
                      0.1 * loss_relation +
                      0.3 * loss_itc +
                      0.5 * loss_itm +
                      0.05 * loss_igt +
                      1.0 * loss_answer)

        return {
            'answer_accuracy': answer_accuracy,
            'loss_answer': loss_answer,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'loss_igt': loss_igt,
            'loss_object': loss_object,
            'loss_relation': loss_relation,
            'total_loss': total_loss,
            'answer_predictions': torch.sigmoid(answer_logits).detach(),
            'answer_labels': answer_labels.detach(),
            'object_confidence': object_confidence.detach(),
            'spatial_info': spatial_info.detach(),
        }