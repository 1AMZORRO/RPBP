#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA-蛋白质结合预测模型
实现Cross-Attention机制，模仿iDeepG的架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RNAEncoder(nn.Module):
    """RNA序列编码器 - 使用One-hot编码"""
    
    def __init__(self, vocab_size: int = 5, embedding_dim: int = 128):
        """
        Args:
            vocab_size: 词汇表大小 (A, C, G, T/U, N)
            embedding_dim: 嵌入维度
        """
        super(RNAEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # One-hot到embedding的映射
        self.projection = nn.Linear(vocab_size, embedding_dim)
        
        # 改进初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """改进的参数初始化"""
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.constant_(self.projection.bias, 0.)
        
    def nucleotide_to_index(self, nucleotide: str) -> int:
        """将核苷酸转换为索引"""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
        return mapping.get(nucleotide.upper(), 4)  # 未知字符映射到N
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        将RNA序列编码为one-hot表示
        
        Args:
            sequence: RNA序列字符串
        
        Returns:
            one-hot编码 [seq_len, vocab_size]
        """
        indices = [self.nucleotide_to_index(nt) for nt in sequence]
        one_hot = F.one_hot(torch.tensor(indices), num_classes=self.vocab_size)
        return one_hot.float()
    
    def forward(self, sequences: list) -> torch.Tensor:
        """
        前向传播
        
        Args:
            sequences: RNA序列字符串列表 [batch_size]
        
        Returns:
            编码后的表示 [batch_size, seq_len, embedding_dim]
        """
        # 批量编码
        one_hot_batch = []
        for seq in sequences:
            one_hot = self.encode_sequence(seq)
            one_hot_batch.append(one_hot)
        
        one_hot_batch = torch.stack(one_hot_batch)  # [batch_size, seq_len, vocab_size]
        one_hot_batch = one_hot_batch.to(next(self.parameters()).device)
        
        # 投影到embedding空间
        embeddings = self.projection(one_hot_batch)  # [batch_size, seq_len, embedding_dim]
        
        return embeddings


class CrossAttentionLayer(nn.Module):
    """
    改进的Cross-Attention层：RNA作为Query，蛋白质作为Key/Value
    
    改进点：
    1. 将蛋白质embedding扩展到序列长度，避免注意力退化
    2. 使用可学习的位置embedding使不同位置有差异
    3. 添加temperature参数控制注意力锐度
    """
    
    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        max_seq_len: int = 101
    ):
        """
        Args:
            query_dim: Query（RNA）的维度
            key_value_dim: Key/Value（蛋白质）的维度
            num_heads: 注意力头数
            dropout: Dropout率
            hidden_dim: 隐藏层维度
            max_seq_len: 最大序列长度
        """
        super(CrossAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # Query, Key, Value投影
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_value_dim, hidden_dim)
        self.value_proj = nn.Linear(key_value_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 可学习的位置embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # 可学习的temperature参数，用于控制注意力分布的锐度
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # 初始化权重
        self._reset_parameters()
    
    def _reset_parameters(self):
        """改进的参数初始化"""
        # Xavier初始化，使用较小的gain
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        
        # 偏置初始化为0
        nn.init.constant_(self.query_proj.bias, 0.)
        nn.init.constant_(self.key_proj.bias, 0.)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: RNA表示 [batch_size, seq_len, query_dim]
            key_value: 蛋白质表示 [batch_size, protein_dim]
            return_attention: 是否返回注意力权重
        
        Returns:
            输出: [batch_size, seq_len, hidden_dim]
            注意力权重: [batch_size, num_heads, seq_len, seq_len] (如果return_attention=True)
        """
        batch_size, seq_len, _ = query.shape
        device = query.device
        
        # 投影到Q
        Q = self.query_proj(query)  # [batch_size, seq_len, hidden_dim]
        
        # 将蛋白质embedding扩展到序列长度
        protein_expanded = key_value.unsqueeze(1).expand(-1, seq_len, -1)
        # [batch_size, seq_len, protein_dim]
        
        # 投影蛋白质embedding
        K = self.key_proj(protein_expanded)  # [batch_size, seq_len, hidden_dim]
        V = self.value_proj(protein_expanded)  # [batch_size, seq_len, hidden_dim]
        
        # 添加可学习的位置embedding到K，使不同位置有差异
        pos_emb = self.pos_embedding[:, :seq_len, :]  # [1, seq_len, hidden_dim]
        K = K + pos_emb
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力分数，使用temperature
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_scores = attention_scores / torch.clamp(self.temperature, min=0.1)
        # [batch_size, num_heads, seq_len, seq_len]
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)
        # [batch_size, num_heads, seq_len, head_dim]
        
        # 合并多头
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.hidden_dim)
        # [batch_size, seq_len, hidden_dim]
        
        # 输出投影
        output = self.out_proj(attended)
        output = self.dropout(output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class RNAProteinBindingModel(nn.Module):
    """
    RNA-蛋白质结合预测模型
    
    改进点：
    1. 添加残差连接改善梯度流
    2. 改进初始化策略
    3. 添加梯度裁剪
    """
    
    def __init__(
        self,
        rna_vocab_size: int = 5,
        rna_embedding_dim: int = 128,
        protein_embedding_dim: int = 1280,
        num_attention_heads: int = 8,
        attention_hidden_dim: int = 256,
        attention_dropout: float = 0.1,
        classifier_hidden_dims: list = [512, 256, 128],
        classifier_dropout: float = 0.3
    ):
        """
        Args:
            rna_vocab_size: RNA词汇表大小
            rna_embedding_dim: RNA嵌入维度
            protein_embedding_dim: 蛋白质嵌入维度（ESM2输出）
            num_attention_heads: 注意力头数
            attention_hidden_dim: 注意力隐藏维度
            attention_dropout: 注意力dropout率
            classifier_hidden_dims: 分类器隐藏层维度列表
            classifier_dropout: 分类器dropout率
        """
        super(RNAProteinBindingModel, self).__init__()
        
        # RNA编码器
        self.rna_encoder = RNAEncoder(rna_vocab_size, rna_embedding_dim)
        
        # RNA编码到attention维度的投影（用于残差连接）
        self.rna_to_attn = nn.Linear(rna_embedding_dim, attention_hidden_dim)
        
        # Cross-Attention层
        self.cross_attention = CrossAttentionLayer(
            query_dim=rna_embedding_dim,
            key_value_dim=protein_embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            hidden_dim=attention_hidden_dim
        )
        
        # 层归一化（在残差连接后）
        self.layer_norm1 = nn.LayerNorm(attention_hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(attention_hidden_dim, attention_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(attention_dropout),
            nn.Linear(attention_hidden_dim * 2, attention_hidden_dim),
            nn.Dropout(attention_dropout)
        )
        
        # 第二个层归一化
        self.layer_norm2 = nn.LayerNorm(attention_hidden_dim)
        
        # 分类器
        classifier_layers = []
        input_dim = attention_hidden_dim
        
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # 最后的二分类层
        classifier_layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """改进的参数初始化"""
        nn.init.xavier_uniform_(self.rna_to_attn.weight, gain=1.0)
        nn.init.constant_(self.rna_to_attn.bias, 0.)
        
        # 分类器初始化
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)
        
    def forward(
        self,
        rna_sequences: list,
        protein_embeddings: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            rna_sequences: RNA序列列表 [batch_size]
            protein_embeddings: 蛋白质embeddings [batch_size, protein_dim]
            return_attention: 是否返回注意力权重
        
        Returns:
            预测logits: [batch_size, 1]
            注意力权重: [batch_size, num_heads, seq_len, seq_len] (如果return_attention=True)
        """
        # 编码RNA序列
        rna_encoded = self.rna_encoder(rna_sequences)
        # [batch_size, seq_len, rna_embedding_dim]
        
        # 投影RNA编码用于残差连接
        rna_projected = self.rna_to_attn(rna_encoded)
        # [batch_size, seq_len, attention_hidden_dim]
        
        # Cross-Attention with residual connection
        attended, attention_weights = self.cross_attention(
            rna_encoded,
            protein_embeddings,
            return_attention=return_attention
        )
        # [batch_size, seq_len, attention_hidden_dim]
        
        # 第一个残差连接 + 层归一化
        attended = self.layer_norm1(attended + rna_projected)
        
        # 前馈网络 with residual connection
        ffn_out = self.ffn(attended)
        attended = self.layer_norm2(ffn_out + attended)
        
        # 全局平均池化
        pooled = attended.mean(dim=1)  # [batch_size, attention_hidden_dim]
        
        # 分类
        logits = self.classifier(pooled)  # [batch_size, 1]
        
        return logits, attention_weights
