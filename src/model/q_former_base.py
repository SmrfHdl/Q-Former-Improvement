import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from layers.cross_modal_transformer import CrossModalTransformer
from model.clip_vit import VisionEncoder
from loguru import logger


class QFormerBase(nn.Module):
    """
    Q-Former architecture with improved regularization and learnable temperature.
    """
    def __init__(
            self, 
            sequence_size: int,
            qformer_hidden_size: int,
            blocks_num: int,
            num_heads: int,
            num_queries: int,
            device: torch.device,
            use_clip_for_text: bool = True,
            clip_model_name: str = "openai/clip-vit-large-patch14",
            dropout_rate: float = 0.3,
            unfreeze_layers: int = 0,
            # New parameters for v2 improvements
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
        super(QFormerBase, self).__init__()

        self.vision_dim = 1024 # ViT default

        self.device = device
        self.max_text_len = sequence_size
        self.use_clip_for_text = use_clip_for_text
        self.unfreeze_layers = unfreeze_layers
        
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
        
        # Increased dropout for regularization (v2 improvement)
        self.vision_dropout = nn.Dropout(dropout_rate)
        self.text_dropout = nn.Dropout(dropout_rate)
        
        # Additional feature dropout for better regularization
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

        # v2 IMPROVEMENT: Learnable temperature for ITC loss
        # This allows the model to find the optimal temperature during training
        if learnable_temperature:
            # Initialize log_temperature so that exp(log_temp) = initial_temperature
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature)))
            logger.info(f"Using LEARNABLE temperature, initial value: {initial_temperature}")
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(initial_temperature)))
            logger.info(f"Using FIXED temperature: {initial_temperature}")
        
        self.learnable_temperature = learnable_temperature

        self.itm_head = nn.Sequential(
            nn.Linear(qformer_hidden_size, qformer_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(qformer_hidden_size // 2, 2)
        )

        self.lm_head = nn.Linear(qformer_hidden_size, self.tokenizer.vocab_size)

        # Answer head with strong regularization to prevent overfitting
        self.answer_head = nn.Sequential(
            nn.Dropout(0.5),  # Heavy dropout BEFORE first layer
            nn.Linear(qformer_hidden_size, qformer_hidden_size // 4),  # Smaller capacity
            nn.GELU(),
            nn.Dropout(0.5),  # Heavy dropout
            nn.Linear(qformer_hidden_size // 4, 1)
        )

        self.cat_mlp = nn.Sequential(
            nn.Linear(qformer_hidden_size * 2, qformer_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(qformer_hidden_size, 1)
        )

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

    def init_weights(self):
        # Use Xavier uniform initialization for projection layers (standard for linear projections)
        # No aggressive scaling - let LayerNorm handle normalization
        nn.init.xavier_uniform_(self.vision_projection.weight)
        nn.init.zeros_(self.vision_projection.bias)

        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)

        nn.init.normal_(self.learned_queries, std=0.02)

        # Initialize itm_head (Sequential)
        for layer in self.itm_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

        # Initialize answer_head (Sequential)
        for layer in self.answer_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for layer in self.cat_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        final_layer = self.cat_mlp[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.02)
            final_layer.bias.data.fill_(0.0)

    def _unfreeze_clip_layers(self, unfreeze_layers: int):
        for param in self.clip_model.parameters():
            param.requires_grad = False

        for i, block in enumerate(reversed(self.clip_model.text_model.encoder.layers)):
            if i < unfreeze_layers:
                for param in block.parameters():
                    param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze {unfreeze_layers} layers of CLIP text model. Trainable parameters: {trainable_params}")

    def _unfreeze_bert_layers(self, unfreeze_layers: int):
        for param in self.bert.parameters():
            param.requires_grad = False

        for i, layer in enumerate(reversed(self.bert.encoder.layer)):
            if i < unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True

        # for param in self.bert.pooler.parameters():
        #     param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze {unfreeze_layers} layers of BERT model. Trainable parameters: {trainable_params}")

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
        """
        Generates attention mask based on the task (ITM, IGT, ITC) and padding mask.

        Args:
            task (str): Task type - 'itm', 'igt', or 'itc'.
            query_len (int): number of tokens in the query.
            pad_mask (torch.Tensor): 1: valid token (not padding), 0: padding token. 
                Shape: (batch_size, seq_len)
            device (torch.device): Device to place the tensors on.

        Returns:
            Attention mask (torch.Tensor): 0 means "can attend", large negative value means "cannot attend".
                Shape: batch_size, total_len, total_len
        """
        batch_size, text_len = pad_mask.size()
        total_len = query_len + text_len 

        # Use a large finite negative value instead of -inf to avoid NaN in softmax backward
        # -inf causes NaN when entire rows are masked: softmax([-inf, -inf, ...]) = 0/0 = NaN
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

        padding_postitions = (pad_mask == 0)

        for b in range(batch_size):
            if padding_postitions[b].any():
                pad_indices = torch.nonzero(padding_postitions[b], as_tuple=True)[0]

                task_mask[b, :, query_len + pad_indices] = MASK_VALUE

                task_mask[b, query_len + pad_indices, :] = MASK_VALUE

        return task_mask
    
    def forward(self, samples: dict):
        image_input = samples['image_input']
        image_features = self.vision_encoder.encode(image_input) # (batch_size, num_patches = 256, hidden_dim = 1024)

        question = samples['question']
        
        batch_size = image_features.shape[0]

        # Project and normalize vision features
        image_features = self.vision_projection(image_features)
        image_features = self.vision_norm(image_features)
        image_features = self.vision_dropout(image_features)
        # Check for NaN/Inf after vision projection
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            logger.warning(f"WARNING: NaN/Inf detected in image_features after projection")
            logger.warning(f"Vision projection weight stats: min={self.vision_projection.weight.min().item():.4f}, max={self.vision_projection.weight.max().item():.4f}")
        # NOTE: Removed L2 normalization here - it was hurting gradient flow through cross-attention
        # LayerNorm already provides sufficient normalization

        queries = self.learned_queries.expand(image_features.shape[0], -1, -1).clone() # (batch_size, num_queries = 32, hidden_dim = 768)

        question_output, question_tokens = self.encode_text(question)
        
        # FIX: Project ALL text embeddings to same space as image_features
        # This ensures both modalities are in the same representation space for the transformer
        text_embeddings_raw = question_output['last_hidden_state']  # (batch_size, seq_len, text_dim)
        text_embeddings_projected = self.text_projection(text_embeddings_raw)  # Project all tokens
        text_embeddings_projected = self.text_norm(text_embeddings_projected)
        text_embeddings_projected = self.text_dropout(text_embeddings_projected)

        # Image Text Contrastive - use EOS token position (CLIP uses EOS, not position 0!)
        # For CLIP: EOS token is at the position of the last non-padding token
        # We find it using argmax on input_ids to locate EOS token (id=49407 for CLIP)
        if self.use_clip_for_text:
            # CLIP EOS token id is 49407, find its position in each sequence
            eos_token_id = 49407
            # Find position of EOS token (first occurrence of EOS in each sequence)
            eos_positions = (question_tokens['input_ids'] == eos_token_id).int().argmax(dim=-1)
            # Gather the embedding at EOS position for each batch item
            batch_indices = torch.arange(text_embeddings_projected.size(0), device=self.device)
            cls_text_embedding = text_embeddings_projected[batch_indices, eos_positions, :]
        else:
            # For BERT, use CLS token at position 0
            cls_text_embedding = text_embeddings_projected[:, 0, :]  # (batch_size, hidden_dim)
        
        # L2 normalize for ITC similarity computation
        cls_text_embedding_normalized = F.normalize(cls_text_embedding, p=2, dim=-1, eps=1e-6)

        attention_mask = self.generate_attention_mask(
            task='itc',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        ) # (batch_size, num_queries + text_len, num_queries + text_len)
        
        # FIX: Pass PROJECTED text_embeddings to transformer (same space as image)
        queries, _ = self.cross_modal_transformer(
            queries,           # queries as Q in cross-attention
            image_features,    # image_features as K,V in cross-attention  
            text_embeddings=text_embeddings_projected,  # PROJECTED text for self-attention
            attention_mask=attention_mask
        )

        # Check for NaN/Inf after cross-modal transformer
        if torch.isnan(queries).any() or torch.isinf(queries).any():
            logger.warning(f"WARNING: NaN/Inf detected in queries after cross_modal_transformer")
            logger.warning(f"queries stats: min={queries.min().item():.4f}, max={queries.max().item():.4f}, mean={queries.mean().item():.4f}")

        # Keep un-normalized queries for answer prediction (better gradient flow)
        queries_for_answer = queries
        
        # L2 normalize ONLY for ITC similarity computation
        queries_normalized = F.normalize(queries, p=2, dim=-1, eps=1e-6)

        # Image to Text similarity calculation (use normalized queries and text)
        # queries_normalized: (batch_size, num_queries, dim), cls_text_embedding_normalized: (batch_size, dim)
        # We want: (batch_size, batch_size, num_queries)
        sim_i2t = torch.einsum("bqd, Bd -> bBq", queries_normalized, cls_text_embedding_normalized)

        sim_i2t, _ = sim_i2t.max(-1) # Max over queries: (b, B) where b=query batch, B=text batch
        
        # v2 IMPROVEMENT: Use learnable temperature with clamping for stability
        temperature = torch.exp(self.log_temperature)
        temperature = torch.clamp(temperature, min=self.temperature_min, max=self.temperature_max)
        
        # Check for NaN/Inf in similarity matrix
        if torch.isnan(sim_i2t).any() or torch.isinf(sim_i2t).any():
            logger.warning(f"WARNING: NaN/Inf detected in sim_i2t before temperature scaling")
            logger.warning(f"sim_i2t stats: min={sim_i2t.min().item():.4f}, max={sim_i2t.max().item():.4f}")
            logger.warning(f"temperature value: {temperature.item():.6f}")
        
        # Clamp similarity before temperature scaling to prevent overflow
        sim_i2t = torch.clamp(sim_i2t, min=-100, max=100)
        
        # Apply temperature scaling
        sim_i2t = sim_i2t / temperature

        sim_t2i = sim_i2t.T # (batch_size, q, t)

        targets = torch.arange(batch_size, device=image_features.device, dtype=int)

        # v2 IMPROVEMENT: Use configurable label smoothing
        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=self.label_smoothing_itc) +
                    F.cross_entropy(sim_t2i, targets, label_smoothing=self.label_smoothing_itc)) / 2
        
        # Image Text Matching
        with torch.no_grad():
            sim_i2t_clone = sim_i2t.clone()
            sim_t2i_clone = sim_t2i.clone()
            sim_i2t_clone.fill_diagonal_(-10000)
            sim_t2i_clone.fill_diagonal_(-10000)
            
            # Clamp values to prevent inf/nan in softmax
            sim_i2t_clone = torch.clamp(sim_i2t_clone, min=-100, max=100)
            sim_t2i_clone = torch.clamp(sim_t2i_clone, min=-100, max=100)

        weights_t2i = torch.softmax(sim_t2i_clone, dim=-1)
        weights_i2t = torch.softmax(sim_i2t_clone, dim=-1)
        
        # Replace any NaN or Inf with uniform distribution
        weights_t2i = torch.where(torch.isnan(weights_t2i) | torch.isinf(weights_t2i), 
                                  torch.ones_like(weights_t2i) / batch_size, 
                                  weights_t2i)
        weights_i2t = torch.where(torch.isnan(weights_i2t) | torch.isinf(weights_i2t), 
                                  torch.ones_like(weights_i2t) / batch_size, 
                                  weights_i2t)
        
        # Ensure weights sum to 1 and are non-negative (add epsilon instead of clamping)
        weights_t2i = weights_t2i + 1e-8
        weights_i2t = weights_i2t + 1e-8
        weights_t2i = weights_t2i / weights_t2i.sum(dim=-1, keepdim=True)
        weights_i2t = weights_i2t / weights_i2t.sum(dim=-1, keepdim=True)

        image_embeddings_negative = []
        for b in range(batch_size):
            try:
                negative_idx = torch.multinomial(weights_t2i[b], 1).item()
            except RuntimeError:
                # If multinomial fails, use argmax as fallback
                negative_idx = torch.argmax(weights_t2i[b]).item()
            image_embeddings_negative.append(image_features[negative_idx])

        image_embeddings_negative = torch.stack(image_embeddings_negative, dim=0) # (batch_size, num_patches, hidden_dim)

        text_embeddings_negative = []
        attention_masks_negative = []

        for b in range(batch_size):
            try:
                negative_idx = torch.multinomial(weights_i2t[b], 1).item()
            except RuntimeError:
                # If multinomial fails, use argmax as fallback
                negative_idx = torch.argmax(weights_i2t[b]).item()
            # FIX: Use projected text embeddings for consistency
            text_embeddings_negative.append(text_embeddings_projected[negative_idx])
            attention_masks_negative.append(question_tokens['attention_mask'][negative_idx])

        text_embeddings_negative = torch.stack(text_embeddings_negative, dim=0) # (batch_size, max_len, hidden_dim)
        attention_masks_negative = torch.stack(attention_masks_negative, dim=0) # (batch_size, max_len)
        attention_masks_negative = self.generate_attention_mask(
            task='itm',
            query_len=queries.shape[1],
            pad_mask=attention_masks_negative,
            device=self.device
        )

        # ============ BLIP-style ITM ============
        # Key insight: Reuse queries from ITC for positive samples, only compute new for negatives
        
        # For NEGATIVE pairs: need to run cross_modal_transformer with wrong pairs
        # Negative type 1: (wrong_image, correct_text)
        attention_mask_neg1 = self.generate_attention_mask(
            task='itm',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        queries_neg1 = self.learned_queries.expand(batch_size, -1, -1).clone()
        queries_neg1, _ = self.cross_modal_transformer(
            queries_neg1,
            image_embeddings_negative,  # wrong image
            text_embeddings=text_embeddings_projected,  # correct text
            attention_mask=attention_mask_neg1
        )
        
        # Negative type 2: (correct_image, wrong_text)  
        queries_neg2 = self.learned_queries.expand(batch_size, -1, -1).clone()
        queries_neg2, _ = self.cross_modal_transformer(
            queries_neg2,
            image_features,  # correct image
            text_embeddings=text_embeddings_negative,  # wrong text
            attention_mask=attention_masks_negative
        )
        
        # BLIP-style: Use FIRST query as [CLS] token for ITM prediction
        # Positive: use queries from ITC (already computed above)
        queries_pos_cls = queries[:, 0, :]  # (batch_size, dim) - reuse ITC queries!
        queries_neg1_cls = queries_neg1[:, 0, :]  # (batch_size, dim)
        queries_neg2_cls = queries_neg2[:, 0, :]  # (batch_size, dim)
        
        # Stack all [CLS] representations
        itm_cls_features = torch.cat([queries_pos_cls, queries_neg1_cls, queries_neg2_cls], dim=0)
        
        # ITM head on [CLS] token only
        logits = self.itm_head[0](itm_cls_features)  # First linear
        logits = self.itm_head[1](logits)  # GELU
        logits = self.itm_head[2](logits)  # Dropout
        logits = self.itm_head[3](logits)  # Final linear -> (batch_size * 3, 2)

        itm_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long),   # Positive: (image, text) - matched
            torch.zeros(batch_size, dtype=torch.long),  # Negative: (wrong_image, text)
            torch.zeros(batch_size, dtype=torch.long)   # Negative: (image, wrong_text)
        ], dim=0).to(self.device)

        # Use class weights for 1:2 imbalance
        itm_class_weights = torch.tensor([1.0, 2.0], device=self.device)
        
        loss_itm = F.cross_entropy(logits, itm_labels, weight=itm_class_weights,
                                   label_smoothing=self.label_smoothing_itm)
        
        # Calculate ITM accuracy
        itm_predictions = torch.argmax(logits, dim=-1)
        itm_accuracy = (itm_predictions == itm_labels).float().mean()

        # Image Grounded Text Generation (IGT)
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
        
        # FIX: Project IGT text embeddings to same space as image
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
            queries_igt,     # queries first
            image_features,  # image second
            text_embeddings=igt_text_projected,  # FIX: Use projected text
            attention_mask=igt_attention_mask
        )

        text_logits = self.lm_head(text_embeddings_igt) # (batch_size, seq_len, vocab_size)

        # Language modeling loss (shifted)
        # Shift the logits and labels (predict next token)
        shifted_logits = text_logits[:, :-1, :] 
        shifted_labels = labels[:, 1:]

        # Always calculate loss, but it will be small if most labels are masked
        loss_igt = F.cross_entropy(
            shifted_logits.reshape(-1, self.tokenizer.vocab_size),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction='mean'
        )

        # Answer Prediction - Use queries from cross_modal_transformer
        # The queries have attended to both image and text, capturing multimodal info
        # queries_for_answer shape: (batch_size, num_queries, dim)
        
        # Max pool over queries to get the most relevant query representation
        max_pooled_queries = torch.max(queries_for_answer, dim=1)[0]  # (batch_size, dim)
        
        # Use answer_head for prediction
        answer_logits = self.answer_head(max_pooled_queries)

        answers = samples['answer']

        dict = {'yes': 1, 'no': 0}

        answers_labels = torch.tensor([
            dict[answer] for answer in answers], dtype=torch.float, device=answer_logits.device).unsqueeze(1)
        
        # v2 IMPROVEMENT: Label smoothing for binary classification
        # Smooth labels: 0 -> label_smoothing/2, 1 -> 1 - label_smoothing/2
        if self.label_smoothing_answer > 0:
            smoothed_labels = answers_labels * (1 - self.label_smoothing_answer) + self.label_smoothing_answer / 2
        else:
            smoothed_labels = answers_labels
            
        loss_answer = F.binary_cross_entropy_with_logits(
            answer_logits,
            smoothed_labels
        )

        p = torch.sigmoid(answer_logits)

        # Calculate accuracy directly (use original labels for accuracy)
        predictions = (p > 0.5).float()
        answer_accuracy = (predictions == answers_labels).float().mean()

        # v2 IMPROVEMENT: Use configurable loss weights
        # IGT is disabled by default (0.0) as it causes severe overfitting
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
            'answer_predictions': p.detach(),
            'answer_labels': answers_labels.detach(),
            'temperature': temperature.detach(),  # Log current temperature
        }