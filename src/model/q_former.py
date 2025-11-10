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


class QFormer(nn.Module):
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
        super(QFormer, self).__init__()

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
    
    def _setup_bert_model(self, unfreeze_bert_layers: int):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
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

        nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)

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
            nn.init.normal_(final_layer, mean=0.0, std=0.01)
            final_layer.bias.data.fill_(0.0)

        logger.info("Model weights initialized.")

    def _unfreeze_clip_layers(self, num_layers: int):
        for param in self.clip_model.parameters():
            param.requires_grad = False

        for i, block in enumerate(reversed(self.clip_model.text_model.encoder.layers)):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze {num_layers} layers of CLIP text model. Trainable parameters: {trainable}")

    def _unfreeze_bert_layers(self, num_layers: int):
        for param in self.bert.parameters():
            param.requires_grad = False

        for i, block in enumerate(reversed(self.bert.encoder.layer)):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze {num_layers} layers of BERT model. Trainable parameters: {trainable}")

    def encode_text(self, questions):
        if self.use_clip_for_text:
            question_tokens = self.clip_processor(
                text=questions,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            ).to(self.device)

            question_tokens = {key: val.to(self.device) for key, val in question_tokens.items()}

            question_output = self.clip_model.text_model(
                input_ids=question_tokens['input_ids'],
                attention_mask=question_tokens['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
        else:
            question_tokens = self.tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            ).to(self.device)

            question_tokens = {key: val.to(self.device) for key, val in question_tokens.items()}

            question_output = self.bert(
                input_ids=question_tokens['input_ids'],
                attention_mask=question_tokens['attention_mask'],
                return_dict=True,
            )

        return question_output, question_tokens
    
    def generate_attention_mask(self, task: str, query_len: int, pad_mask: torch.Tensor, device: torch.device):
        """
        Generates an attention mask based on the task (itm, igt, itc) and padding mask.

        Args:
            task: 
        """