import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from layers.cross_modal_transformer import CrossModalTransformer
from model.clip_vit import VisionEncoder
from loguru import logger
import math

# SCENE GRAPH GENERATION (SGG) COMPONENTS

class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer with Edge Attention.
    Implements message passing between object nodes weighted by relation importance.
    
    Based on: Graph Attention Networks (GAT) + Relational GCN concepts
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Node feature transformations
        self.W_node = nn.Linear(dim, dim)
        
        # Edge attention - computes attention weights for each edge
        self.W_edge = nn.Sequential(
            nn.Linear(dim * 2 + dim, dim),  # subject + object + edge features
            nn.LeakyReLU(0.2),
            nn.Linear(dim, num_heads)
        )
        
        # Message transformation
        self.W_msg = nn.Linear(dim, dim)
        
        # Edge feature update
        self.W_edge_update = nn.Sequential(
            nn.Linear(dim * 3, dim),  # subject + object + edge
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, 
                adjacency_weights: torch.Tensor = None):
        """
        Args:
            node_features: (batch, num_nodes, dim)
            edge_features: (batch, num_nodes, num_nodes, dim)
            adjacency_weights: Optional pre-computed weights (batch, num_nodes, num_nodes)
        Returns:
            updated_nodes: (batch, num_nodes, dim)
            updated_edges: (batch, num_nodes, num_nodes, dim)
        """
        batch, num_nodes, dim = node_features.shape
        residual = node_features
        
        # Transform node features
        h = self.W_node(node_features)  # (B, N, D)
        
        # Compute edge attention scores
        # Create pairwise node representations
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # (B, N, N, D)
        
        # Concatenate subject, object, and edge features
        edge_input = torch.cat([h_i, h_j, edge_features], dim=-1)  # (B, N, N, 3D)
        edge_attn = self.W_edge(edge_input)  # (B, N, N, num_heads)
        
        # Apply softmax per node (normalize incoming messages)
        edge_attn = F.softmax(edge_attn, dim=2)  # Normalize over source nodes
        edge_attn = self.dropout(edge_attn)
        
        if adjacency_weights is not None:
            edge_attn = edge_attn * adjacency_weights.unsqueeze(-1)
        
        # Compute messages
        messages = self.W_msg(h_j)  # (B, N, N, D)
        
        # Aggregate messages with attention
        # Average across heads
        edge_attn_mean = edge_attn.mean(dim=-1, keepdim=True)  # (B, N, N, 1)
        aggregated = (messages * edge_attn_mean).sum(dim=2)  # (B, N, D)
        
        # Output projection with residual
        updated_nodes = self.norm(residual + self.dropout(self.out_proj(aggregated)))
        
        # Update edge features
        edge_update_input = torch.cat([h_i, h_j, edge_features], dim=-1)
        updated_edges = edge_features + self.W_edge_update(edge_update_input)
        
        return updated_nodes, updated_edges


class SpatialRelationEncoder(nn.Module):
    """
    Encodes spatial relationships between objects using relative position, size, and overlap.
    More sophisticated than simple bbox concatenation.
    """
    def __init__(self, dim: int, num_spatial_features: int = 16):
        super().__init__()
        self.dim = dim
        
        # Spatial feature extractors
        # Each pair generates: relative position (4), relative size (2), IoU-like (1), 
        # angle (2), distance (1), overlap ratios (2), aspect ratios (2), center offset (2)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(num_spatial_features, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.LayerNorm(dim)
        )
        
    def compute_spatial_features(self, boxes: torch.Tensor):
        """
        Compute rich spatial features between all pairs of boxes.
        
        Args:
            boxes: (batch, num_objects, 4) normalized boxes [x, y, w, h]
        Returns:
            spatial_features: (batch, num_objects, num_objects, 16)
        """
        batch, num_obj, _ = boxes.shape
        
        # Extract box components
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        
        # Compute centers and areas
        cx = x + w / 2
        cy = y + h / 2
        areas = w * h + 1e-6
        
        # Create pairwise features
        # Centers
        cx_i = cx.unsqueeze(2)  # (B, N, 1)
        cy_i = cy.unsqueeze(2)
        cx_j = cx.unsqueeze(1)  # (B, 1, N)
        cy_j = cy.unsqueeze(1)
        
        # Relative center offset
        dx = cx_j - cx_i  # (B, N, N)
        dy = cy_j - cy_i
        
        # Distance between centers
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        
        # Angle between centers
        angle = torch.atan2(dy, dx)
        angle_sin = torch.sin(angle)
        angle_cos = torch.cos(angle)
        
        # Sizes
        w_i, h_i = w.unsqueeze(2), h.unsqueeze(2)
        w_j, h_j = w.unsqueeze(1), h.unsqueeze(1)
        
        # Relative size ratios
        w_ratio = torch.log(w_j / (w_i + 1e-6) + 1e-6)
        h_ratio = torch.log(h_j / (h_i + 1e-6) + 1e-6)
        
        # Area ratio
        area_i = areas.unsqueeze(2)
        area_j = areas.unsqueeze(1)
        area_ratio = torch.log(area_j / (area_i + 1e-6) + 1e-6)
        
        # Aspect ratios
        aspect_i = (w_i / (h_i + 1e-6)).clamp(0.1, 10)
        aspect_j = (w_j / (h_j + 1e-6)).clamp(0.1, 10)
        aspect_diff = torch.log(aspect_j / (aspect_i + 1e-6) + 1e-6)
        
        # Compute IoU-like overlap
        x1_i, y1_i = x.unsqueeze(2), y.unsqueeze(2)
        x2_i, y2_i = (x + w).unsqueeze(2), (y + h).unsqueeze(2)
        x1_j, y1_j = x.unsqueeze(1), y.unsqueeze(1)
        x2_j, y2_j = (x + w).unsqueeze(1), (y + h).unsqueeze(1)
        
        inter_x1 = torch.max(x1_i, x1_j)
        inter_y1 = torch.max(y1_i, y1_j)
        inter_x2 = torch.min(x2_i, x2_j)
        inter_y2 = torch.min(y2_i, y2_j)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        union_area = area_i + area_j - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        # Containment ratios
        contain_ij = inter_area / (area_j + 1e-6)  # How much of j is in i
        contain_ji = inter_area / (area_i + 1e-6)  # How much of i is in j
        
        # Stack all features: 16 features total
        spatial_features = torch.stack([
            dx, dy,                           # 2: relative position
            dist,                             # 1: distance
            angle_sin, angle_cos,             # 2: angle
            w_ratio, h_ratio,                 # 2: relative size
            area_ratio,                       # 1: area ratio
            aspect_diff,                      # 1: aspect ratio difference
            iou,                              # 1: IoU
            contain_ij, contain_ji,           # 2: containment
            torch.log(aspect_i.squeeze(2).unsqueeze(1).expand(-1, num_obj, -1) + 1e-6),  # 1
            torch.log(aspect_j.squeeze(1).unsqueeze(2).expand(-1, -1, num_obj) + 1e-6),  # 1
            torch.ones_like(dist),            # 1: bias term
        ], dim=-1)  # (B, N, N, 16)
        
        return spatial_features
    
    def forward(self, boxes: torch.Tensor):
        """
        Args:
            boxes: (batch, num_objects, 4)
        Returns:
            spatial_embeddings: (batch, num_objects, num_objects, dim)
        """
        spatial_features = self.compute_spatial_features(boxes)
        return self.spatial_mlp(spatial_features)


class RelationTypePredictor(nn.Module):
    """
    Predicts relation types with semantic categories.
    Inspired by Neural Motifs and VCTree for scene graph generation.
    """
    def __init__(self, dim: int, num_relation_types: int = 16, dropout: float = 0.1):
        super().__init__()
        
        # Relation type categories:
        # Spatial: above, below, left, right, in front, behind, inside, outside
        # Semantic: has, holds, wears, uses, near, part of
        # Action: looking at, interacting with
        
        self.relation_types = num_relation_types
        
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_relation_types)
        )
        
        # Type embeddings for different relation categories
        self.type_embeddings = nn.Embedding(num_relation_types, dim)
        
    def forward(self, edge_features: torch.Tensor):
        """
        Args:
            edge_features: (batch, num_obj, num_obj, dim)
        Returns:
            relation_logits: (batch, num_obj, num_obj, num_types)
            relation_embeddings: (batch, num_obj, num_obj, dim) - weighted type embeddings
        """
        logits = self.predictor(edge_features)
        probs = F.softmax(logits, dim=-1)
        
        # Compute weighted type embeddings
        # probs: (B, N, N, T), type_embeddings: (T, D)
        relation_embeddings = torch.einsum('bnmt,td->bnmd', probs, self.type_embeddings.weight)
        
        return logits, relation_embeddings


class SceneGraphGenerator(nn.Module):
    """
    Full Scene Graph Generation module.
    
    Architecture:
    1. Initialize node and edge features
    2. Multiple rounds of message passing via Graph Convolution
    3. Predict relation types
    4. Output enriched node and edge representations
    
    Inspired by: Graph R-CNN, Neural Motifs, KERN
    """
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 3, 
                 num_relation_types: int = 16, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_layers = num_layers
        
        # Spatial relation encoder
        self.spatial_encoder = SpatialRelationEncoder(dim)
        
        # Semantic edge initialization (from visual features)
        self.semantic_edge_init = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norms between graph layers
        self.node_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        self.edge_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        
        # Relation type predictor
        self.relation_predictor = RelationTypePredictor(dim, num_relation_types, dropout)
        
        # Text-guided relation refinement
        self.text_relation_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Output projection
        self.node_output = nn.Linear(dim, dim)
        self.edge_output = nn.Linear(dim, dim)
        
    def forward(self, object_features: torch.Tensor, spatial_info: torch.Tensor,
                text_embeddings: torch.Tensor = None, attention_mask: torch.Tensor = None):
        """
        Args:
            object_features: (batch, num_objects, dim)
            spatial_info: (batch, num_objects, 4) - bounding boxes
            text_embeddings: Optional text features for guidance (batch, seq_len, dim)
            attention_mask: Optional attention mask
            
        Returns:
            enriched_nodes: (batch, num_objects, dim)
            edge_features: (batch, num_objects, num_objects, dim)
            relation_logits: (batch, num_objects, num_objects, num_types)
        """
        batch, num_obj, dim = object_features.shape
        
        # Initialize edge features
        # 1. Spatial component
        spatial_edges = self.spatial_encoder(spatial_info)  # (B, N, N, D)
        
        # 2. Semantic component from pairwise object features
        obj_i = object_features.unsqueeze(2).expand(-1, -1, num_obj, -1)
        obj_j = object_features.unsqueeze(1).expand(-1, num_obj, -1, -1)
        semantic_edges = self.semantic_edge_init(
            torch.cat([obj_i, obj_j], dim=-1)
        )  # (B, N, N, D)
        
        # Combine spatial and semantic
        edge_features = spatial_edges + semantic_edges
        node_features = object_features
        
        # Message passing rounds
        for i, graph_layer in enumerate(self.graph_layers):
            # Apply graph convolution
            node_features, edge_features = graph_layer(node_features, edge_features)
            
            # Normalize
            node_features = self.node_norms[i](node_features)
            edge_features = self.edge_norms[i](edge_features)
        
        # Text-guided relation refinement (if text provided)
        if text_embeddings is not None:
            # Flatten edges for attention
            batch, n, m, d = edge_features.shape
            edges_flat = edge_features.reshape(batch, n * m, d)
            
            # Cross-attention with text
            refined_edges, _ = self.text_relation_attn(
                edges_flat, text_embeddings, text_embeddings,
                key_padding_mask=(attention_mask == 0) if attention_mask is not None else None
            )
            
            # Reshape and add residual
            edge_features = edge_features + refined_edges.reshape(batch, n, m, d)
        
        # Predict relation types
        relation_logits, relation_embeddings = self.relation_predictor(edge_features)
        
        # Enhance edge features with relation type information
        edge_features = edge_features + relation_embeddings
        
        # Final output projections
        enriched_nodes = self.node_output(node_features)
        enriched_edges = self.edge_output(edge_features)
        
        return enriched_nodes, enriched_edges, relation_logits


# NEURAL STATE MACHINE (NSM) FOR HIERARCHICAL REASONING

class ControlUnit(nn.Module):
    """
    Control Unit for Neural State Machine.
    Attends to the question to determine the current reasoning operation.
    
    Inspired by MAC Network's Control Unit.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        
        # Question attention - selects relevant part of question
        self.question_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Control state update
        self.control_update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Control word projection
        self.control_proj = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, prev_control: torch.Tensor, question_embeddings: torch.Tensor, 
                question_mask: torch.Tensor = None):
        """
        Args:
            prev_control: Previous control state (batch, dim)
            question_embeddings: Question features (batch, seq_len, dim)
            question_mask: Mask for question (batch, seq_len)
            
        Returns:
            control: Updated control state (batch, dim)
        """
        batch = prev_control.shape[0]
        
        # Attend to question based on previous control
        query = prev_control.unsqueeze(1)  # (B, 1, D)
        
        key_padding_mask = None
        if question_mask is not None:
            key_padding_mask = (question_mask == 0)
        
        attended_question, _ = self.question_attn(
            query, question_embeddings, question_embeddings,
            key_padding_mask=key_padding_mask
        )  # (B, 1, D)
        attended_question = attended_question.squeeze(1)  # (B, D)
        
        # Update control state
        control_input = torch.cat([prev_control, attended_question], dim=-1)
        control = self.control_update(control_input)
        control = self.norm(control + self.control_proj(attended_question))
        
        return control


class ReadUnit(nn.Module):
    """
    Read Unit for Neural State Machine.
    Reads from the knowledge base (object/relation features) based on control signal.
    
    Implements hierarchical attention over objects and their relations.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Control-guided attention over objects
        self.object_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Control-guided attention over relations
        self.relation_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Combine object and relation information
        self.combine = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Memory interaction
        self.memory_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, control: torch.Tensor, object_features: torch.Tensor,
                relation_features: torch.Tensor, memory: torch.Tensor):
        """
        Args:
            control: Control state (batch, dim)
            object_features: Object representations (batch, num_obj, dim)
            relation_features: Relation representations (batch, num_relations, dim)
            memory: Previous memory state (batch, dim)
            
        Returns:
            read_output: Information retrieved (batch, dim)
        """
        batch = control.shape[0]
        
        # Control as query
        query = control.unsqueeze(1)  # (B, 1, D)
        
        # Attend to objects
        obj_info, obj_attn = self.object_attn(query, object_features, object_features)
        obj_info = obj_info.squeeze(1)  # (B, D)
        
        # Attend to relations
        rel_info, rel_attn = self.relation_attn(query, relation_features, relation_features)
        rel_info = rel_info.squeeze(1)  # (B, D)
        
        # Combine object and relation information
        combined = self.combine(torch.cat([obj_info, rel_info], dim=-1))
        
        # Gate with memory
        gate_input = torch.cat([combined, memory], dim=-1)
        gate = self.memory_gate(gate_input)
        
        read_output = self.norm(gate * combined + (1 - gate) * memory)
        
        return read_output, obj_attn, rel_attn


class WriteUnit(nn.Module):
    """
    Write Unit for Neural State Machine.
    Updates the memory based on read output and control signal.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention gate for memory update
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )
        
        # Memory update transformation
        self.memory_transform = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, memory: torch.Tensor, read_output: torch.Tensor, control: torch.Tensor):
        """
        Args:
            memory: Current memory state (batch, dim)
            read_output: Read unit output (batch, dim)
            control: Control signal (batch, dim)
            
        Returns:
            new_memory: Updated memory (batch, dim)
        """
        # Compute gate
        gate_input = torch.cat([memory, read_output, control], dim=-1)
        gate = self.gate(gate_input)
        
        # Compute update
        update_input = torch.cat([read_output, control], dim=-1)
        update = self.memory_transform(update_input)
        
        # Apply gated update
        new_memory = self.norm(gate * update + (1 - gate) * memory)
        
        return new_memory


class ReasoningCell(nn.Module):
    """
    Single reasoning step combining Control, Read, and Write units.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.control_unit = ControlUnit(dim, num_heads, dropout)
        self.read_unit = ReadUnit(dim, num_heads, dropout)
        self.write_unit = WriteUnit(dim, dropout)
        
    def forward(self, control: torch.Tensor, memory: torch.Tensor,
                question_embeddings: torch.Tensor, object_features: torch.Tensor,
                relation_features: torch.Tensor, question_mask: torch.Tensor = None):
        """
        Args:
            control: Previous control state (batch, dim)
            memory: Previous memory state (batch, dim)
            question_embeddings: Question features (batch, seq_len, dim)
            object_features: Object representations (batch, num_obj, dim)
            relation_features: Relation representations (batch, num_rel, dim)
            question_mask: Mask for question
            
        Returns:
            new_control, new_memory, attention_weights
        """
        # Update control
        new_control = self.control_unit(control, question_embeddings, question_mask)
        
        # Read from knowledge base
        read_output, obj_attn, rel_attn = self.read_unit(
            new_control, object_features, relation_features, memory
        )
        
        # Write to memory
        new_memory = self.write_unit(memory, read_output, new_control)
        
        return new_control, new_memory, {'obj_attn': obj_attn, 'rel_attn': rel_attn}


class NeuralStateMachine(nn.Module):
    """
    Neural State Machine for Multi-hop Hierarchical Reasoning.
    
    Performs iterative reasoning steps:
    1. Control: What to look for next based on question
    2. Read: Retrieve relevant object/relation information
    3. Write: Update memory with new information
    4. Repeat for multiple hops
    
    Inspired by: MAC Network, Neural Module Networks, Neural State Machines for VQA
    """
    def __init__(self, dim: int, num_heads: int = 8, num_hops: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_hops = num_hops
        
        # Initial control state (learned)
        self.init_control = nn.Parameter(torch.randn(1, dim))
        
        # Initial memory state (learned)
        self.init_memory = nn.Parameter(torch.randn(1, dim))
        
        # Reasoning cells (shared or different per hop)
        # Using shared weights (like original MAC) for efficiency
        self.reasoning_cell = ReasoningCell(dim, num_heads, dropout)
        
        # Hop-specific position encodings
        self.hop_embeddings = nn.Parameter(torch.randn(num_hops, dim))
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),  # memory + control
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, question_embeddings: torch.Tensor, object_features: torch.Tensor,
                relation_features: torch.Tensor, question_mask: torch.Tensor = None):
        """
        Args:
            question_embeddings: Question features (batch, seq_len, dim)
            object_features: Object representations (batch, num_obj, dim)
            relation_features: Relation representations (batch, num_rel, dim)
            question_mask: Mask for question
            
        Returns:
            final_output: Reasoning result (batch, dim)
            memory_states: All memory states for visualization (num_hops, batch, dim)
            attention_weights: Attention weights from all hops
        """
        batch = question_embeddings.shape[0]
        
        # Initialize control and memory
        control = self.init_control.expand(batch, -1)
        memory = self.init_memory.expand(batch, -1)
        
        # Store states for analysis
        memory_states = []
        all_attention = []
        
        # Multi-hop reasoning
        for hop in range(self.num_hops):
            # Add hop-specific encoding
            hop_encoding = self.hop_embeddings[hop].unsqueeze(0).expand(batch, -1)
            control_input = control + hop_encoding
            
            # Reasoning step
            control, memory, attn_weights = self.reasoning_cell(
                control_input, memory,
                question_embeddings, object_features, relation_features,
                question_mask
            )
            
            memory_states.append(memory)
            all_attention.append(attn_weights)
        
        # Combine final control and memory for output
        final_output = self.output_proj(torch.cat([control, memory], dim=-1))
        final_output = self.norm(final_output)
        
        return final_output, torch.stack(memory_states), all_attention


# HIERARCHICAL REASONING PATH (Level 3)

class HierarchicalReasoningPath(nn.Module):
    """
    Level 3: Hierarchical Reasoning with Neural State Machine.
    
    Features:
    1. Neural State Machine for multi-hop compositional reasoning
    2. Hierarchical integration of object and relation information
    3. Task-specific decoders for VQA, ITM, and text generation
    """
    def __init__(self, dim: int, num_heads: int = 8, num_hops: int = 4,
                 num_global_queries: int = 32, vocab_size: int = 49408, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        
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
        
        # Answer prediction head
        self.answer_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
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
        if attention_mask is not None:
            # Extract just the text part of the mask
            query_len = self.global_queries.shape[1]
            question_mask = None  # Will be handled by NSM
        else:
            question_mask = None
        
        # Neural State Machine reasoning
        nsm_output, memory_states, nsm_attention = self.nsm(
            text_embeddings, object_features, relation_features, question_mask
        )  # nsm_output: (batch, dim)
        
        # Global context via transformer
        global_queries = self.global_queries.expand(batch, -1, -1).clone()
        
        # Combine hierarchical features for cross-attention
        hierarchical_features = torch.cat([object_features, relation_features], dim=1)
        
        global_features, updated_text = self.global_transformer(
            global_queries, hierarchical_features, text_embeddings, attention_mask
        )
        
        # Pool global features
        global_pooled = global_features.mean(dim=1)  # (batch, dim)
        
        # Fuse NSM output with global features
        fused = self.fusion(torch.cat([nsm_output, global_pooled], dim=-1))
        fused = self.norm(fused)
        
        # Task predictions
        itm_logits = self.itm_head(fused)
        lm_logits = self.lm_head(updated_text)
        answer_logits = self.answer_head(fused)
        
        return global_features, itm_logits, lm_logits, answer_logits, nsm_attention


# IMPROVED OBJECT DETECTION PATH (Level 1)

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
        combined = torch.cat([lower_level, current_level], dim=-1)
        gate_values = self.gate(combined)
        return gate_values * lower_level + (1 - gate_values) * current_level


class ObjectDetectionPath(nn.Module):
    """
    Level 1: Object Detection Path
    Extracts object-level features with spatial and attribute information.
    """
    def __init__(self, dim: int, num_heads: int, num_layers: int, 
                 num_object_queries: int, dropout: float = 0.1):
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
        
        # Object feature refinement
        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, image_features: torch.Tensor, text_embeddings: torch.Tensor, 
                attention_mask: torch.Tensor = None):
        batch_size = image_features.shape[0]
        object_queries = self.object_queries.expand(batch_size, -1, -1).clone()
        
        # Extract object features through cross-attention with image
        object_features, _ = self.object_transformer(
            object_queries, image_features, text_embeddings, attention_mask
        )
        
        # Predict spatial information and confidence
        spatial_info = torch.sigmoid(self.spatial_head(object_features))
        confidence_logits = self.confidence_head(object_features)
        
        # Refine object features
        object_features = self.refine(object_features)
        
        return object_features, spatial_info, confidence_logits


# MAIN Q-FORMER IMPROVED MODEL

class QFormerImproved(nn.Module):
    """
    Improved Q-Former with 3-Level Hierarchical Multi-Path Reasoning.
    
    Architecture:
        Level 1: Object Detection Path - Extracts object-level features
        Level 2: Scene Graph Generation - Builds relational graph with GNN
        Level 3: Neural State Machine - Multi-hop compositional reasoning
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
        
        # Level 3: Hierarchical Reasoning with NSM
        self.reasoning_path = HierarchicalReasoningPath(
            dim=qformer_hidden_size,
            num_heads=num_heads,
            num_hops=num_reasoning_hops,
            num_global_queries=num_global_queries,
            vocab_size=self.tokenizer.vocab_size if hasattr(self, 'tokenizer') else 49408,
            dropout=dropout_rate
        )

        # ITC temperature (fixed buffer)
        self.register_buffer('temperature', torch.tensor(0.07))

        self.init_weights()
        self.to(device)
        
        # Log model architecture
        self._log_model_info()

    def _log_model_info(self):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"QFormerImproved initialized:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Level 1: Object Detection Path")
        logger.info(f"  - Level 2: Scene Graph Generation (GNN with Message Passing)")
        logger.info(f"  - Level 3: Neural State Machine (Multi-hop Reasoning)")

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
        """Generate attention mask based on task type."""
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
        Forward pass through 3-level hierarchical architecture.
        
        Level 1: Object Detection
        Level 2: Scene Graph Generation (GNN)
        Level 3: Neural State Machine Reasoning
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
        
        # Auxiliary loss for object detection
        loss_object = F.binary_cross_entropy_with_logits(
            object_confidence_logits.squeeze(-1),
            torch.ones_like(object_confidence_logits.squeeze(-1)) * 0.5
        )
        
        object_confidence = torch.sigmoid(object_confidence_logits)

        # === LEVEL 2: Scene Graph Generation ===
        # Build scene graph with GNN message passing
        enriched_objects, edge_features, relation_logits = self.scene_graph(
            object_features, spatial_info, text_embeddings, 
            question_tokens['attention_mask']
        )
        
        # Flatten edge features to relation sequence for Level 3
        num_obj = enriched_objects.shape[1]
        relation_features = edge_features.reshape(batch_size, num_obj * num_obj, -1)
        
        # Top-k most confident relations (reduce computation for Level 3)
        with torch.no_grad():
            relation_importance = relation_logits.max(dim=-1)[0]  # (B, N, N)
            relation_importance = relation_importance.reshape(batch_size, -1)
            k = min(64, relation_importance.shape[1])  # Top 64 relations
            _, top_indices = relation_importance.topk(k, dim=1)
        
        # Gather top relations
        relation_features = torch.gather(
            relation_features, 1, 
            top_indices.unsqueeze(-1).expand(-1, -1, relation_features.shape[-1])
        )
        
        # Relation classification loss (soft supervision)
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

        # === Image-Text Matching (ITM) Loss with hard negatives ===
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
        
        # Run reasoning path for negative pairs
        neg_enriched_objects = enriched_objects[neg_image_indices]
        neg_relation_features = relation_features[torch.tensor(neg_text_indices, device=self.device)]
        neg_image_features = image_features[neg_image_indices]
        neg_text_embeddings = text_embeddings[torch.tensor(neg_text_indices, device=self.device)]
        
        # Negative 1: wrong image + correct text
        _, itm_logits_neg1, _, _, _ = self.reasoning_path(
            enriched_objects, relation_features, neg_image_features, text_embeddings, attention_mask_l3
        )
        
        # Negative 2: correct image + wrong text  
        _, itm_logits_neg2, _, _, _ = self.reasoning_path(
            enriched_objects, relation_features, image_features, neg_text_embeddings, attention_mask_l3
        )
        
        # Combine ITM logits
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

        # === Answer Prediction Loss ===
        answers = samples['answer']
        answer_dict = {'yes': 1, 'no': 0}
        answer_labels = torch.tensor(
            [answer_dict[ans] for ans in answers],
            dtype=torch.float, device=self.device
        ).unsqueeze(1)
        
        label_smoothing = 0.1
        smoothed_labels = answer_labels * (1 - label_smoothing) + label_smoothing / 2
        loss_answer = F.binary_cross_entropy_with_logits(answer_logits, smoothed_labels)
        
        predictions = (torch.sigmoid(answer_logits) > 0.5).float()
        answer_accuracy = (predictions == answer_labels).float().mean()

        # === Total Loss with Hierarchical Weighting ===
        total_loss = (0.05 * loss_object +
                      0.1 * loss_relation +  # Higher weight for SGG
                      1.0 * loss_itc +
                      1.0 * loss_itm +
                      0.0 * loss_igt +  # Disabled for VQA
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
            'answer_predictions': torch.sigmoid(answer_logits).detach(),
            'answer_labels': answer_labels.detach(),
            'object_confidence': object_confidence.detach(),
            'spatial_info': spatial_info.detach(),
            'relation_logits': relation_logits.detach(),
            'nsm_attention': nsm_attention,  # For visualization
        }
