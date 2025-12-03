"""
Q-Former Base for Generative Open-Ended VQA.

This model GENERATES answer text instead of classifying into fixed vocabulary.
- Training: Condition on image + question, generate answer text
- Inference: Auto-regressive generation of answer
- Evaluation: Compare generated text with ground truth (exact/soft match)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from layers.cross_modal_transformer import CrossModalTransformer
from model.clip_vit import VisionEncoder
from loguru import logger


class QFormerBaseGenerative(nn.Module):
    """
    Q-Former Base for Generative VQA.
    
    Key features:
    - Uses LM head to generate answer text
    - Training: Teacher forcing with answer tokens
    - Inference: Auto-regressive generation
    - Evaluation: Exact match / soft match with ground truth
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
            max_answer_length: int = 10,
            learnable_temperature: bool = True,
            initial_temperature: float = 0.07,
            temperature_min: float = 0.01,
            temperature_max: float = 0.5,
            label_smoothing_itc: float = 0.1,
            label_smoothing_itm: float = 0.1,
            loss_weight_itc: float = 0.2,
            loss_weight_itm: float = 0.3,
            loss_weight_generation: float = 1.0,
            stochastic_depth_rate: float = 0.1):
        super(QFormerBaseGenerative, self).__init__()

        self.vision_dim = 1024  # ViT default
        self.device = device
        self.max_text_len = sequence_size
        self.max_answer_length = max_answer_length
        self.use_clip_for_text = use_clip_for_text
        self.unfreeze_layers = unfreeze_layers
        self.qformer_hidden_size = qformer_hidden_size
        
        # Loss weights
        self.label_smoothing_itc = label_smoothing_itc
        self.label_smoothing_itm = label_smoothing_itm
        self.loss_weight_itc = loss_weight_itc
        self.loss_weight_itm = loss_weight_itm
        self.loss_weight_generation = loss_weight_generation
        
        # Temperature bounds
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
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(initial_temperature)))
        self.learnable_temperature = learnable_temperature

        # ITM head
        self.itm_head = nn.Sequential(
            nn.Linear(qformer_hidden_size, qformer_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(qformer_hidden_size // 2, 2)
        )

        # LM head for answer generation
        self.lm_head = nn.Linear(qformer_hidden_size, self.tokenizer.vocab_size)

        # Special tokens
        self.answer_start_token = "[ANS]"
        self.answer_end_token = "[/ANS]"
        
        # Add special tokens to tokenizer
        self._add_special_tokens()

        self.init_weights()
        self.to(device)
        
        logger.info(f"QFormerBaseGenerative initialized")
        logger.info(f"  - Max answer length: {max_answer_length}")
        logger.info(f"  - Vocab size: {self.tokenizer.vocab_size}")

    def _add_special_tokens(self):
        """Add special tokens for answer generation."""
        special_tokens = {
            'additional_special_tokens': [self.answer_start_token, self.answer_end_token, "[DEC]"]
        }
        
        if self.use_clip_for_text:
            self.clip_tokenizer.add_special_tokens(special_tokens)
            new_vocab_size = len(self.clip_tokenizer)
            old_vocab_size = self.clip_model.text_model.embeddings.token_embedding.weight.shape[0]
            
            if new_vocab_size != old_vocab_size:
                old_embeddings = self.clip_model.text_model.embeddings.token_embedding
                embedding_dim = old_embeddings.embedding_dim
                new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
                new_embeddings.weight.data[:old_vocab_size] = old_embeddings.weight.data
                self.clip_model.text_model.embeddings.token_embedding = new_embeddings
            
            self.ans_start_id = self.clip_tokenizer.convert_tokens_to_ids(self.answer_start_token)
            self.ans_end_id = self.clip_tokenizer.convert_tokens_to_ids(self.answer_end_token)
            self.dec_token_id = self.clip_tokenizer.convert_tokens_to_ids("[DEC]")
            
            # Update LM head for new vocab size
            self.lm_head = nn.Linear(self.qformer_hidden_size, new_vocab_size).to(self.device)
        else:
            self.tokenizer.add_special_tokens(special_tokens)
            self.bert.resize_token_embeddings(len(self.tokenizer))
            
            self.ans_start_id = self.tokenizer.convert_tokens_to_ids(self.answer_start_token)
            self.ans_end_id = self.tokenizer.convert_tokens_to_ids(self.answer_end_token)
            self.dec_token_id = self.tokenizer.convert_tokens_to_ids("[DEC]")
            
            self.lm_head = nn.Linear(self.qformer_hidden_size, len(self.tokenizer)).to(self.device)

    def _setup_clip_model(self, clip_model_name: str, unfreeze_clip_layers: int):
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_tokenizer = self.clip_processor.tokenizer

        for param in self.clip_model.parameters():
            param.requires_grad = False

        if unfreeze_clip_layers > 0:
            self._unfreeze_clip_layers(unfreeze_clip_layers)

        self.text_dim = self.clip_model.text_model.config.hidden_size
        self.tokenizer = self.clip_tokenizer

    def _setup_bert_model(self, unfreeze_bert_layers: int):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device)

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

    def _unfreeze_bert_layers(self, unfreeze_layers: int):
        for i, layer in enumerate(reversed(self.bert.encoder.layer)):
            if i < unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True

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

    def encode_text(self, texts: list[str]):
        if self.use_clip_for_text:
            tokens = self.clip_processor(
                text=texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            output = self.clip_model.text_model(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                output_hidden_states=True
            )
        else:
            tokens = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            output = self.bert(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                return_dict=True
            )
        return output, tokens
    
    def generate_attention_mask(self, task: str, query_len: int, pad_mask: torch.Tensor, device: torch.device = 'cpu'):
        batch_size, text_len = pad_mask.size()
        total_len = query_len + text_len 
        MASK_VALUE = -1e9
        task_mask = torch.zeros((batch_size, total_len, total_len), device=device)

        if task == 'itm':
            pass
        elif task == "generation":
            # Causal mask for generation
            causal_indices = torch.triu_indices(text_len, text_len, offset=1, device=device)
            for b in range(batch_size):
                task_mask[b, query_len + causal_indices[0], query_len + causal_indices[1]] = MASK_VALUE
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
    
    def prepare_answer_targets(self, answers: list[str]):
        """
        Prepare answer tokens for training.
        Format: [ANS] answer_text [/ANS]
        """
        # Format answers with special tokens
        formatted_answers = [
            f"{self.answer_start_token} {ans.lower().strip()} {self.answer_end_token}"
            for ans in answers
        ]
        
        # Tokenize
        if self.use_clip_for_text:
            tokens = self.clip_tokenizer(
                formatted_answers,
                padding='max_length',
                truncation=True,
                max_length=self.max_answer_length + 2,  # +2 for special tokens
                return_tensors='pt'
            )
        else:
            tokens = self.tokenizer(
                formatted_answers,
                padding='max_length',
                truncation=True,
                max_length=self.max_answer_length + 2,
                return_tensors='pt'
            )
        
        return tokens['input_ids'].to(self.device), tokens['attention_mask'].to(self.device)

    def forward(self, samples: dict):
        """
        Forward pass for generative VQA.
        
        Training mode:
        - Input: image + question + answer (teacher forcing)
        - Output: loss for answer generation
        
        Args:
            samples: Dict with keys:
                - image_input: Dict with 'pixel_values'
                - question: List of question strings
                - answer: List of answer strings (ground truth)
        """
        image_input = samples['image_input']
        questions = samples['question']
        answers = samples['answer']
        batch_size = len(questions)

        # Encode image
        image_features = self.vision_encoder.encode(image_input)
        image_features = self.vision_projection(image_features)
        image_features = self.vision_norm(image_features)
        image_features = self.vision_dropout(image_features)

        # Encode question
        question_output, question_tokens = self.encode_text(questions)
        text_embeddings = question_output['last_hidden_state']
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = self.text_norm(text_embeddings)
        text_embeddings = self.text_dropout(text_embeddings)

        queries = self.learned_queries.expand(batch_size, -1, -1).clone()

        # ITC
        if self.use_clip_for_text:
            eos_token_id = 49407
            eos_positions = (question_tokens['input_ids'] == eos_token_id).int().argmax(dim=-1)
            batch_indices = torch.arange(batch_size, device=self.device)
            cls_text_embedding = text_embeddings[batch_indices, eos_positions, :]
        else:
            cls_text_embedding = text_embeddings[:, 0, :]
        
        cls_text_embedding = F.normalize(cls_text_embedding, p=2, dim=-1, eps=1e-6)

        attention_mask_itc = self.generate_attention_mask(
            task='itc',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        queries_itc, _ = self.cross_modal_transformer(
            queries, image_features, text_embeddings=text_embeddings, attention_mask=attention_mask_itc
        )

        queries_normalized = F.normalize(queries_itc, p=2, dim=-1, eps=1e-6)
        sim_i2t = torch.einsum("bqd, Bd -> bBq", queries_normalized, cls_text_embedding)
        sim_i2t, _ = sim_i2t.max(-1)
        
        temperature = torch.exp(self.log_temperature)
        temperature = torch.clamp(temperature, min=self.temperature_min, max=self.temperature_max)
        
        sim_i2t = torch.clamp(sim_i2t, min=-100, max=100) / temperature
        sim_t2i = sim_i2t.T

        targets = torch.arange(batch_size, device=self.device, dtype=int)
        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=self.label_smoothing_itc) +
                    F.cross_entropy(sim_t2i, targets, label_smoothing=self.label_smoothing_itc)) / 2

        # ITM with hard negatives (simplified)
        with torch.no_grad():
            sim_clone = sim_i2t.clone()
            sim_clone.fill_diagonal_(-10000)
        weights = F.softmax(sim_clone, dim=-1)
        weights = torch.where(torch.isnan(weights), torch.ones_like(weights) / batch_size, weights)

        neg_indices = []
        for b in range(batch_size):
            try:
                neg_idx = torch.multinomial(weights[b], 1).item()
            except:
                neg_idx = (b + 1) % batch_size
            neg_indices.append(neg_idx)

        image_neg = image_features[neg_indices]
        
        queries_neg = self.learned_queries.expand(batch_size, -1, -1).clone()
        attention_mask_itm = self.generate_attention_mask(
            task='itm', query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'], device=self.device
        )
        queries_neg, _ = self.cross_modal_transformer(
            queries_neg, image_neg, text_embeddings=text_embeddings, attention_mask=attention_mask_itm
        )

        itm_pos = self.itm_head(queries_itc[:, 0, :])
        itm_neg = self.itm_head(queries_neg[:, 0, :])
        
        itm_logits = torch.cat([itm_pos, itm_neg], dim=0)
        itm_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long, device=self.device),
            torch.zeros(batch_size, dtype=torch.long, device=self.device)
        ])
        
        loss_itm = F.cross_entropy(itm_logits, itm_labels, label_smoothing=self.label_smoothing_itm)
        itm_accuracy = (torch.argmax(itm_logits, dim=-1) == itm_labels).float().mean()

        # === ANSWER GENERATION (Teacher Forcing) ===
        # Prepare input: [DEC] question [ANS] answer_tokens
        # Target: shift by 1 for next token prediction
        
        # Create combined input for generation
        generation_inputs = [
            f"[DEC] {q} {self.answer_start_token} {a.lower().strip()}"
            for q, a in zip(questions, answers)
        ]
        
        if self.use_clip_for_text:
            gen_tokens = self.clip_tokenizer(
                generation_inputs,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len + self.max_answer_length,
                return_tensors='pt'
            )
        else:
            gen_tokens = self.tokenizer(
                generation_inputs,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len + self.max_answer_length,
                return_tensors='pt'
            )
        
        gen_input_ids = gen_tokens['input_ids'].to(self.device)
        gen_attention_mask = gen_tokens['attention_mask'].to(self.device)
        
        # Encode for generation
        if self.use_clip_for_text:
            gen_output = self.clip_model.text_model(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask
            )
        else:
            gen_output = self.bert(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                return_dict=True
            )
        
        gen_embeddings = self.text_projection(gen_output['last_hidden_state'])
        gen_embeddings = self.text_norm(gen_embeddings)
        
        # Cross-modal generation
        queries_gen = self.learned_queries.expand(batch_size, -1, -1).clone()
        attention_mask_gen = self.generate_attention_mask(
            task='generation',
            query_len=queries_gen.shape[1],
            pad_mask=gen_attention_mask,
            device=self.device
        )
        
        _, gen_text_out = self.cross_modal_transformer(
            queries_gen, image_features, text_embeddings=gen_embeddings, attention_mask=attention_mask_gen
        )
        
        # LM logits
        lm_logits = self.lm_head(gen_text_out)  # (B, seq_len, vocab_size)
        
        # Create labels (shift input_ids by 1)
        labels = gen_input_ids.clone()
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        
        # Compute generation loss (shifted)
        shifted_logits = lm_logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        
        loss_generation = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100
        )

        # Total loss
        total_loss = (self.loss_weight_itc * loss_itc +
                      self.loss_weight_itm * loss_itm +
                      self.loss_weight_generation * loss_generation)

        # Generate answers for accuracy calculation (greedy)
        with torch.no_grad():
            generated_answers = self.generate(
                image_features, questions, max_length=self.max_answer_length
            )
            
            # Calculate exact match accuracy
            correct = sum(
                1 for gen, gt in zip(generated_answers, answers)
                if gen.lower().strip() == gt.lower().strip()
            )
            answer_accuracy = torch.tensor(correct / batch_size, device=self.device)

        return {
            'answer_accuracy': answer_accuracy,
            'itm_accuracy': itm_accuracy,
            'loss_generation': loss_generation,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'total_loss': total_loss,
            'generated_answers': generated_answers,
            'ground_truth_answers': answers,
            'temperature': temperature.detach(),
        }

    @torch.no_grad()
    def generate(self, image_features: torch.Tensor, questions: list[str], 
                 max_length: int = 10, temperature: float = 1.0) -> list[str]:
        """
        Auto-regressive answer generation.
        
        Args:
            image_features: Encoded image features (B, patches, dim)
            questions: List of question strings
            max_length: Maximum answer length
            temperature: Sampling temperature (1.0 = greedy)
            
        Returns:
            List of generated answer strings
        """
        batch_size = len(questions)
        
        # Start with [DEC] question [ANS]
        prompts = [f"[DEC] {q} {self.answer_start_token}" for q in questions]
        
        if self.use_clip_for_text:
            tokens = self.clip_tokenizer(
                prompts,
                padding=True,
                return_tensors='pt'
            )
        else:
            tokens = self.tokenizer(
                prompts,
                padding=True,
                return_tensors='pt'
            )
        
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        generated_tokens = []
        
        for _ in range(max_length):
            # Encode current sequence
            if self.use_clip_for_text:
                text_output = self.clip_model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            else:
                text_output = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            text_embeddings = self.text_projection(text_output['last_hidden_state'])
            text_embeddings = self.text_norm(text_embeddings)
            
            # Cross-modal processing
            queries = self.learned_queries.expand(batch_size, -1, -1).clone()
            gen_mask = self.generate_attention_mask(
                task='generation',
                query_len=queries.shape[1],
                pad_mask=attention_mask,
                device=self.device
            )
            
            _, text_out = self.cross_modal_transformer(
                queries, image_features, text_embeddings=text_embeddings, attention_mask=gen_mask
            )
            
            # Get next token logits (from last position)
            logits = self.lm_head(text_out[:, -1, :])  # (B, vocab_size)
            
            if temperature != 1.0:
                logits = logits / temperature
            
            # Greedy decoding
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            
            generated_tokens.append(next_tokens)
            
            # Check for end token
            if (next_tokens == self.ans_end_id).all():
                break
            
            # Append to input
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones(batch_size, 1, device=self.device)
            ], dim=1)
        
        # Decode generated tokens
        if generated_tokens:
            all_generated = torch.cat(generated_tokens, dim=1)  # (B, gen_len)
            
            answers = []
            for i in range(batch_size):
                token_ids = all_generated[i].tolist()
                
                # Remove end token if present
                if self.ans_end_id in token_ids:
                    token_ids = token_ids[:token_ids.index(self.ans_end_id)]
                
                # Decode
                if self.use_clip_for_text:
                    answer = self.clip_tokenizer.decode(token_ids, skip_special_tokens=True)
                else:
                    answer = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                
                answers.append(answer.strip())
        else:
            answers = [""] * batch_size
        
        return answers
    
    @torch.no_grad()
    def inference(self, image_input: dict, question: str) -> str:
        """
        Single sample inference.
        
        Args:
            image_input: Dict with 'pixel_values' tensor
            question: Question string
            
        Returns:
            Generated answer string
        """
        image_features = self.vision_encoder.encode(image_input)
        image_features = self.vision_projection(image_features)
        image_features = self.vision_norm(image_features)
        
        answers = self.generate(image_features, [question], max_length=self.max_answer_length)
        return answers[0]

