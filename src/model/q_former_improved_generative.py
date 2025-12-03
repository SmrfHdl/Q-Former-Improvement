"""
Q-Former Improved for Generative Open-Ended VQA.

Combines SGG + NSM architecture with generative answer output.
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
    SceneGraphGenerator,
    NeuralStateMachine,
    ObjectDetectionPath,
)


class GenerativeReasoningPath(nn.Module):
    """
    Level 3: Hierarchical Reasoning with NSM for Generative VQA.
    
    Combines NSM reasoning with text generation capability.
    """
    def __init__(self, dim: int, num_heads: int = 8, num_hops: int = 4,
                 num_global_queries: int = 32, vocab_size: int = 49408,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        
        self.global_queries = nn.Parameter(torch.randn(1, num_global_queries, dim))
        self.nsm = NeuralStateMachine(dim, num_heads, num_hops, dropout)
        
        self.global_transformer = CrossModalTransformer(
            dim=dim, num_heads=num_heads, num_layers=2, dropout=dropout
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # ITM head
        self.itm_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 2)
        )
        
        # LM head for generation
        self.lm_head = nn.Linear(dim, vocab_size)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, object_features, relation_features, image_features, 
                text_embeddings, attention_mask=None):
        batch = image_features.shape[0]
        
        # NSM reasoning
        nsm_output, memory_states, nsm_attention = self.nsm(
            text_embeddings, object_features, relation_features, None
        )
        
        # Global context
        global_queries = self.global_queries.expand(batch, -1, -1).clone()
        hierarchical_features = torch.cat([object_features, relation_features], dim=1)
        
        global_features, updated_text = self.global_transformer(
            global_queries, hierarchical_features, text_embeddings, attention_mask
        )
        
        global_pooled = global_features.mean(dim=1)
        fused = self.fusion(torch.cat([nsm_output, global_pooled], dim=-1))
        fused = self.norm(fused)
        
        itm_logits = self.itm_head(fused)
        lm_logits = self.lm_head(updated_text)
        
        return global_features, itm_logits, lm_logits, fused, nsm_attention


class QFormerImprovedGenerative(nn.Module):
    """
    Improved Q-Former for Generative VQA.
    
    Architecture:
        Level 1: Object Detection Path
        Level 2: Scene Graph Generation (GNN)
        Level 3: NSM Reasoning + Text Generation
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
            num_reasoning_hops: int = 4,
            num_relation_types: int = 16,
            max_answer_length: int = 10,
            device: torch.device = torch.device('cuda'),
            use_clip_for_text: bool = True,
            clip_model_name: str = "openai/clip-vit-large-patch14",
            dropout_rate: float = 0.3,
            unfreeze_layers: int = 0):
        super().__init__()

        self.vision_dim = 1024
        self.device = device
        self.max_text_len = sequence_size
        self.max_answer_length = max_answer_length
        self.use_clip_for_text = use_clip_for_text
        self.qformer_hidden_size = qformer_hidden_size

        # Text encoder setup
        if self.use_clip_for_text:
            self._setup_clip_model(clip_model_name, unfreeze_layers)
        else:
            self._setup_bert_model(unfreeze_layers)

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=clip_model_name, device=device, unfreeze_layers=unfreeze_layers
        )
        
        # Projections
        self.vision_projection = nn.Linear(self.vision_dim, qformer_hidden_size)
        self.vision_norm = nn.LayerNorm(qformer_hidden_size)
        self.text_projection = nn.Linear(self.text_dim, qformer_hidden_size)
        self.text_norm = nn.LayerNorm(qformer_hidden_size)
        
        self.vision_dropout = nn.Dropout(dropout_rate * 0.5)
        self.text_dropout = nn.Dropout(dropout_rate * 0.5)

        layers_per_level = max(1, blocks_num // 3)
        
        # Level 1: Object Detection
        self.object_path = ObjectDetectionPath(
            dim=qformer_hidden_size, num_heads=num_heads,
            num_layers=layers_per_level, num_object_queries=num_object_queries,
            dropout=dropout_rate
        )
        
        # Level 2: Scene Graph Generation
        self.scene_graph = SceneGraphGenerator(
            dim=qformer_hidden_size, num_heads=num_heads,
            num_layers=layers_per_level, num_relation_types=num_relation_types,
            dropout=dropout_rate
        )
        
        # Level 3: Generative Reasoning
        self.reasoning_path = GenerativeReasoningPath(
            dim=qformer_hidden_size, num_heads=num_heads,
            num_hops=num_reasoning_hops, num_global_queries=num_global_queries,
            vocab_size=len(self.tokenizer), dropout=dropout_rate
        )

        # Special tokens
        self.answer_start_token = "[ANS]"
        self.answer_end_token = "[/ANS]"
        self._add_special_tokens()

        self.register_buffer('temperature', torch.tensor(0.07))
        self.init_weights()
        self.to(device)
        
        self._log_model_info()

    def _log_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"QFormerImprovedGenerative initialized:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Max answer length: {self.max_answer_length}")

    def _add_special_tokens(self):
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
        else:
            self.tokenizer.add_special_tokens(special_tokens)
            self.bert.resize_token_embeddings(len(self.tokenizer))
            
            self.ans_start_id = self.tokenizer.convert_tokens_to_ids(self.answer_start_token)
            self.ans_end_id = self.tokenizer.convert_tokens_to_ids(self.answer_end_token)
            self.dec_token_id = self.tokenizer.convert_tokens_to_ids("[DEC]")

    def _setup_clip_model(self, clip_model_name: str, unfreeze_clip_layers: int):
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_tokenizer = self.clip_processor.tokenizer

        for param in self.clip_model.parameters():
            param.requires_grad = False

        if unfreeze_clip_layers > 0:
            for i, block in enumerate(reversed(self.clip_model.text_model.encoder.layers)):
                if i < unfreeze_clip_layers:
                    for param in block.parameters():
                        param.requires_grad = True

        self.text_dim = self.clip_model.text_model.config.hidden_size
        self.tokenizer = self.clip_tokenizer

    def _setup_bert_model(self, unfreeze_bert_layers: int):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device)

        for param in self.bert.parameters():
            param.requires_grad = False

        if unfreeze_bert_layers > 0:
            for i, layer in enumerate(reversed(self.bert.encoder.layer)):
                if i < unfreeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.text_dim = self.bert.config.hidden_size

    def init_weights(self):
        nn.init.xavier_uniform_(self.vision_projection.weight)
        nn.init.zeros_(self.vision_projection.bias)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)

    def encode_text(self, texts: list[str]):
        if self.use_clip_for_text:
            tokens = self.clip_processor(
                text=texts, padding='max_length', truncation=True,
                max_length=self.max_text_len, return_tensors='pt'
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            output = self.clip_model.text_model(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                output_hidden_states=True
            )
        else:
            tokens = self.tokenizer(
                texts, padding='max_length', truncation=True,
                max_length=self.max_text_len, return_tensors='pt'
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            output = self.bert(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                return_dict=True
            )
        return output, tokens
    
    def generate_attention_mask(self, task: str, query_len: int, pad_mask: torch.Tensor, device='cpu'):
        batch_size, text_len = pad_mask.size()
        total_len = query_len + text_len 
        MASK_VALUE = -1e9
        task_mask = torch.zeros((batch_size, total_len, total_len), device=device)

        if task == 'itm':
            pass
        elif task == "generation":
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
    
    def forward(self, samples: dict):
        """Forward pass for generative VQA."""
        image_input = samples['image_input']
        questions = samples['question']
        answers = samples['answer']
        batch_size = len(questions)

        # Encode image
        image_features = self.vision_encoder.encode(image_input)
        image_features = self.vision_projection(image_features)
        image_features = self.vision_norm(image_features)
        image_features = self.vision_dropout(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-6)

        # Encode question
        question_output, question_tokens = self.encode_text(questions)
        text_embeddings = question_output['last_hidden_state']
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = self.text_norm(text_embeddings)
        text_embeddings = self.text_dropout(text_embeddings)

        # Level 1: Object Detection
        attention_mask_l1 = self.generate_attention_mask(
            task='itc', query_len=32,
            pad_mask=question_tokens['attention_mask'], device=self.device
        )
        
        object_features, spatial_info, object_confidence_logits = self.object_path(
            image_features, text_embeddings, attention_mask_l1
        )
        
        loss_object = F.binary_cross_entropy_with_logits(
            object_confidence_logits.squeeze(-1),
            torch.ones_like(object_confidence_logits.squeeze(-1)) * 0.5
        )

        # Level 2: Scene Graph
        enriched_objects, edge_features, relation_logits = self.scene_graph(
            object_features, spatial_info, text_embeddings, question_tokens['attention_mask']
        )
        
        num_obj = enriched_objects.shape[1]
        relation_features = edge_features.reshape(batch_size, num_obj * num_obj, -1)
        
        with torch.no_grad():
            relation_importance = relation_logits.max(dim=-1)[0].reshape(batch_size, -1)
            k = min(64, relation_importance.shape[1])
            _, top_indices = relation_importance.topk(k, dim=1)
        
        relation_features = torch.gather(
            relation_features, 1, 
            top_indices.unsqueeze(-1).expand(-1, -1, relation_features.shape[-1])
        )
        
        num_relation_types = relation_logits.shape[-1]
        uniform_target = torch.ones(batch_size, num_obj, num_obj, num_relation_types, device=self.device) / num_relation_types
        loss_relation = F.kl_div(F.log_softmax(relation_logits, dim=-1), uniform_target, reduction='batchmean')

        # Level 3: Generation with reasoning
        # Prepare generation input
        generation_inputs = [
            f"[DEC] {q} {self.answer_start_token} {a.lower().strip()}"
            for q, a in zip(questions, answers)
        ]
        
        if self.use_clip_for_text:
            gen_tokens = self.clip_tokenizer(
                generation_inputs, padding='max_length', truncation=True,
                max_length=self.max_text_len + self.max_answer_length, return_tensors='pt'
            )
        else:
            gen_tokens = self.tokenizer(
                generation_inputs, padding='max_length', truncation=True,
                max_length=self.max_text_len + self.max_answer_length, return_tensors='pt'
            )
        
        gen_input_ids = gen_tokens['input_ids'].to(self.device)
        gen_attention_mask = gen_tokens['attention_mask'].to(self.device)
        
        if self.use_clip_for_text:
            gen_output = self.clip_model.text_model(input_ids=gen_input_ids, attention_mask=gen_attention_mask)
        else:
            gen_output = self.bert(input_ids=gen_input_ids, attention_mask=gen_attention_mask, return_dict=True)
        
        gen_embeddings = self.text_projection(gen_output['last_hidden_state'])
        gen_embeddings = self.text_norm(gen_embeddings)
        
        attention_mask_l3 = self.generate_attention_mask(
            task='generation', query_len=32,
            pad_mask=gen_attention_mask, device=self.device
        )
        
        global_features, itm_logits, lm_logits, fused, nsm_attention = self.reasoning_path(
            enriched_objects, relation_features, image_features, gen_embeddings, attention_mask_l3
        )

        # Generation loss
        labels = gen_input_ids.clone()
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        
        shifted_logits = lm_logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        
        loss_generation = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100
        )

        # ITC Loss
        if self.use_clip_for_text:
            eos_positions = (question_tokens['input_ids'] == 49407).int().argmax(dim=-1)
            batch_indices = torch.arange(batch_size, device=self.device)
            cls_text = question_output['last_hidden_state'][batch_indices, eos_positions, :]
        else:
            cls_text = question_output['last_hidden_state'][:, 0, :]
        cls_text = self.text_projection(cls_text)
        cls_text = F.normalize(cls_text, p=2, dim=-1, eps=1e-6)
        
        global_img = F.normalize(global_features.mean(dim=1), p=2, dim=-1, eps=1e-6)
        
        sim_i2t = torch.clamp(torch.matmul(global_img, cls_text.T), min=-100, max=100) / self.temperature
        sim_t2i = sim_i2t.T
        
        targets = torch.arange(batch_size, device=self.device, dtype=torch.long)
        loss_itc = (F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)) / 2

        # ITM Loss (simplified)
        itm_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
        loss_itm = F.cross_entropy(itm_logits, itm_labels)
        itm_accuracy = (torch.argmax(itm_logits, dim=-1) == itm_labels).float().mean()

        # Total loss
        total_loss = (0.05 * loss_object +
                      0.1 * loss_relation +
                      0.5 * loss_itc +
                      0.5 * loss_itm +
                      2.0 * loss_generation)

        # Generate for accuracy
        with torch.no_grad():
            generated_answers = self.generate(image_features, enriched_objects, relation_features, questions)
            correct = sum(1 for gen, gt in zip(generated_answers, answers) if gen.lower().strip() == gt.lower().strip())
            answer_accuracy = torch.tensor(correct / batch_size, device=self.device)

        return {
            'answer_accuracy': answer_accuracy,
            'itm_accuracy': itm_accuracy,
            'loss_generation': loss_generation,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'loss_object': loss_object,
            'loss_relation': loss_relation,
            'total_loss': total_loss,
            'generated_answers': generated_answers,
            'ground_truth_answers': answers,
        }

    @torch.no_grad()
    def generate(self, image_features, object_features, relation_features, 
                 questions: list[str], max_length: int = None) -> list[str]:
        """Auto-regressive generation with reasoning."""
        if max_length is None:
            max_length = self.max_answer_length
            
        batch_size = len(questions)
        
        prompts = [f"[DEC] {q} {self.answer_start_token}" for q in questions]
        
        if self.use_clip_for_text:
            tokens = self.clip_tokenizer(prompts, padding=True, return_tensors='pt')
        else:
            tokens = self.tokenizer(prompts, padding=True, return_tensors='pt')
        
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        generated_tokens = []
        
        for _ in range(max_length):
            if self.use_clip_for_text:
                text_output = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            
            text_embeddings = self.text_projection(text_output['last_hidden_state'])
            text_embeddings = self.text_norm(text_embeddings)
            
            gen_mask = self.generate_attention_mask(
                task='generation', query_len=32, pad_mask=attention_mask, device=self.device
            )
            
            _, _, lm_logits, _, _ = self.reasoning_path(
                object_features, relation_features, image_features, text_embeddings, gen_mask
            )
            
            logits = lm_logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_tokens)
            
            if (next_tokens == self.ans_end_id).all():
                break
            
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.device)], dim=1)
        
        # Decode
        answers = []
        if generated_tokens:
            all_generated = torch.cat(generated_tokens, dim=1)
            for i in range(batch_size):
                token_ids = all_generated[i].tolist()
                if self.ans_end_id in token_ids:
                    token_ids = token_ids[:token_ids.index(self.ans_end_id)]
                if self.use_clip_for_text:
                    answer = self.clip_tokenizer.decode(token_ids, skip_special_tokens=True)
                else:
                    answer = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                answers.append(answer.strip())
        else:
            answers = [""] * batch_size
        
        return answers

