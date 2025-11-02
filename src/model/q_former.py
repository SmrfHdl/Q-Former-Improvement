import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from layers.cross_modal_transformer import CrossModalTransformer
from torchmetrics import Accuracy
from clip_vit import VisionEncoder


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

    
