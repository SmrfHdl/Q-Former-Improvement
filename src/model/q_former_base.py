import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from layers.cross_modal_transformer import CrossModalTransformer
from torchmetrics import Accuracy
from clip_vit import VisionEncoder
from loguru import logger
from loguru import logger


class QFormerBase(nn.Module):
    """
    Q-Former architecture.
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
            unfreeze_layers: int = 0):
        super(QFormerBase, self).__init__()

        self.vision_dim = 1024 # ViT default

        self.device = device
        self.max_text_len = sequence_size
        self.use_clip_for_text = use_clip_for_text
        self.unfreeze_layers = unfreeze_layers

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
        self.text_projection = nn.Linear(self.text_dim, qformer_hidden_size)

        self.learned_queries = nn.Parameter(
            torch.randn(1, num_queries, qformer_hidden_size)
        )

        self.cross_modal_transformer = CrossModalTransformer(
            qformer_hidden_size,
            num_heads,
            blocks_num,
            dropout=dropout_rate
        )

        self.temperature = nn.Parameter(
            torch.ones([]) * 0.07
        )

        self.itm_head = nn.Linear(qformer_hidden_size, 2)

        self.lm_head = nn.Linear(qformer_hidden_size, self.tokenizer.vocab_size)

        self.answer_head = nn.Linear(qformer_hidden_size, 1)

        self.accuracy = Accuracy(threshold=0.5, num_classes=2, task='binary')

        self.cat_mlp = nn.Sequential(
            nn.Linear(qformer_hidden_size * 2, qformer_hidden_size),
            nn.ReLU(),
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
            old_embeddings = self.clip_model.text_model.embeddings.token_embedding.weight.shape[0]
            new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)

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
        nn.init.kaiming_normal_(self.vision_projection.weight, nonlinearity='relu')
        nn.init.zeros_(self.vision_projection.bias)

        nn.init.kaiming_normal_(self.text_projection.weight, nonlinearity='relu')
        nn.init.zeros_(self.text_projection.bias)

        nn.init.normal_(self.learned_queries, std=0.02)

        nn.init.constant_(self.temperature, 0.07)

        nn.init.normal_(self.itm_head.weight, std=0.02)
        nn.init.zeros_(self.itm_head.bias)

        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

        nn.init.normal_(self.answer_head.weight, std=0.02)
        nn.init.zeros_(self.answer_head.bias)

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
            questions_tokens = self.clip_processor(
                text=questions,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            ).to(self.device)

            questions_tokens = {k: v.to(self.device) for k, v in questions_tokens.items()}

            question_output = self.clip_model.text_model(
                input_ids=questions_tokens['input_ids'],
                attention_mask=questions_tokens['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
        else:
            question_tokens = self.tokenizer(
                questions,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            ).to(self.device)

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
            Attention mask (torch.Tensor): 0 means "can attend", -inf means "cannot attend".
                Shape: batch_size, total_len, total_len
        """
        batch_size, text_len = pad_mask.size()
        total_len = query_len + text_len 

        task_mask = torch.zeros((batch_size, total_len, total_len), device=device)

        if task == 'itm':
            pass

        elif task == "igt":
            causal_indices = torch.triu_indices(text_len, text_len, offset=1, device=device)

            for b in range(batch_size):
                task_mask[b, query_len + causal_indices[0], query_len + causal_indices[1]] = float('-inf')

            task_mask[:, :query_len, query_len:] = float('-inf')

        elif task == 'itc':
            task_mask[:, :query_len, query_len:] = float('-inf')
            task_mask[:, query_len:, :query_len] = float('-inf')

        padding_postitions = (pad_mask == 0)

        for b in range(batch_size):
            if padding_postitions[b].any():
                pad_indices = torch.nonzero(padding_postitions[b], as_tuple=True)[0]

                task_mask[b, :, query_len + pad_indices] = float('-inf')

                task_mask[b, query_len + pad_indices, :] = float('-inf')

        return task_mask
    
    def forward(self, samples: dict):
        image_input = samples['image_input']
        image_features = self.vision_encoder.encode(image_input) # (batch_size, num_patches = 256, hidden_dim = 1024)

        question = samples['question']
        
        batch_size = image_features.shape[0]

        image_features = F.normalize(self.vision_projection(image_features), dim=-1)

        queries = self.learned_queries.expand(image_features.shape[0], -1, -1).clone() # (batch_size, num_queries = 32, hidden_dim = 768)

        question_output, question_tokens = self.encode_text(question)

        # Image Text Contrastive
        cls_text_embedding = question_output['last_hidden_state'][:, 0, :] # (batch_size, hidden_dim)
        cls_text_embedding = F.normalize(self.text_projection(cls_text_embedding), dim=-1) # (batch_size, hidden_dim)

        attention_mask = self.generate_attention_mask(
            task='itc',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        ) # (batch_size, num_queries + text_len, num_queries + text_len)
        queries, _ = self.cross_modal_transformer(
            image_features,
            queries,
            text_embeddings=question_output['last_hidden_state'],
            attention_mask=attention_mask
        )

        queries = F.normalize(queries, dim=-1) # (batch_size, num_queries, hidden_dim)

        # Image to Text similarity calculation
        sim_i2t = torch.einsum("bqd, td -> btq", queries, cls_text_embedding)

        sim_i2t, _ = sim_i2t.max(-1) # Max over queries: (b, t)
        sim_i2t = sim_i2t / self.temperature

        sim_t2i = sim_i2t.T # (batch_size, q, t)

        targets = torch.arange(batch_size, device=image_features.device, dtype=int)

        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
                    F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2
        
        # Image Text Matching
        with torch.no_grad():
            sim_i2t.fill_diagonal_(-10000)
            sim_t2i.fill_diagonal_(-10000)

        weights_t2i = torch.softmax(sim_t2i, dim=-1)
        weights_i2t = torch.softmax(sim_i2t, dim=-1)

        image_embeddings_negative = []
        for b in range(batch_size):
            negative_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeddings_negative.append(image_features[negative_idx])

        image_embeddings_negative = torch.stack(image_embeddings_negative, dim=0) # (batch_size, num_patches, hidden_dim)

        text_embeddings_negative = []
        attention_masks_negative = []

        for b in range(batch_size):
            negative_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeddings_negative.append(question_output['last_hidden_state'][negative_idx])
            attention_masks_negative.append(question_tokens['attention_mask'][negative_idx])

        text_embeddings_negative = torch.stack(text_embeddings_negative, dim=0) # (batch_size, max_len, hidden_dim)
        attention_masks_negative = torch.stack(attention_masks_negative, dim=0) # (batch_size, max_len)
        attention_masks_negative = self.generate_attention_mask(
            task='itm',
            query_len=queries.shape[1],
            pad_mask=attention_masks_negative,
            device=self.device
        )

        text_embeddings_all = torch.cat([
            question_output['last_hidden_state'], question_output['last_hidden_state'], text_embeddings_negative], dim=0)
        
        image_embeddings_all = torch.cat([
            image_features, image_embeddings_negative, image_features], dim=0)
        
        attention_mask_all = torch.cat([
            self.generate_attention_mask(
                task='itm',
                query_len=queries.shape[1],
                pad_mask=question_tokens['attention_mask'],
                device=self.device
            ),
            self.generate_attention_mask(
                task='itm',
                query_len=queries.shape[1],
                pad_mask=question_tokens['attention_mask'],
                device=self.device
            ),
            attention_masks_negative,
        ], dim=0)

        queries_itm = self.learned_queries.expand(image_embeddings_all.shape[0], -1, -1).clone()

        queries_itm, _ = self.cross_modal_transformer(
            image_embeddings_all,
            queries_itm,
            text_embeddings=text_embeddings_all,
            attention_mask=attention_mask_all
        )

        # Perform itm head
        itm_embeddings = self.itm_head(queries_itm) # (batch_size * 3, num_queries, 2)
        logits = torch.mean(itm_embeddings, dim=1) # (batch_size * 3, 2)

        itm_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long)
        ], dim=0).to(self.device)

        loss_itm = F.cross_entropy(logits, itm_labels)

        # Image Grounded Text Generation
        igt_input_ids = question_tokens['input_ids'].clone()
        igt_input_ids[:, 0] = self.dec_token_id

        labels = igt_input_ids.clone()
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        if self.use_clip_for_text:
            igt_text_output = self.clip_model.text_model(
                input_ids=igt_input_ids,
                attention_mask=question_tokens['attention_mask'],
                return_dict=True,
            )
        else:
            igt_text_output = self.bert(
                input_ids=igt_input_ids,
                attention_mask=question_tokens['attention_mask'],
                return_dict=True,
            )
        
        igt_attention_mask = self.generate_attention_mask(
            task='igt',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )

        queries_igt = self.learned_queries.expand(batch_size, -1, -1).clone()

        queries_igt, text_embeddings_igt = self.cross_modal_transformer(
            image_features,
            queries_igt,
            text_embeddings=igt_text_output['last_hidden_state'],
            attention_mask=igt_attention_mask
        )

        text_logits = self.lm_head(text_embeddings_igt) # (batch_size, seq_len, vocab_size)

        # Language modeling loss (shifted)
        # Shift the logits and labels (predict next token)
        shifted_logits = text_logits[:, :-1, :] 
        shifted_labels = labels[:, 1:]

        loss_igt = F.cross_entropy(
            shifted_logits.reshape(-1, self.tokenizer.vocab_size),
            shifted_labels.reshape(-1),
            ignore_index=-100
        )

        # Answer Prediction
        # Queries ITM: (batch_size, 32, 768)

        max_pooled_queries = torch.max(queries_itm[:batch_size], dim=1)[0]

        answer_logits = self.answer_head(max_pooled_queries)

        answers = samples['answer']

        dict = {'yes': 1, 'no': 0}

        answers_labels = torch.tensor([
            dict[answer] for answer in answers], dtype=torch.float, device=answer_logits.device).unsqueeze(1)
        
        loss_answer = F.binary_cross_entropy_with_logits(
            answer_logits,
            answers_labels
        )

        p = torch.sigmoid(answer_logits)

        answer_accuracy = self.accuracy(p, answers_labels.int())

        return {
            'answer_accuracy': answer_accuracy,
            'loss_answer': loss_answer,
            'loss_itc': loss_itc,
            'loss_itm': loss_itm,
            'loss_igt': loss_igt,
            'total_loss': loss_itc + loss_itm + loss_igt + loss_answer,
            'answer_predictions': p.detach(),
            'answer_labels': answers_labels.detach(),
        }