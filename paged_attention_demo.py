"""
Paged Attention Demonstration

This implementation demonstrates the PagedAttention mechanism as described in the paper
"PagedAttention: Efficient Memory Management for Large Language Model Inference"

PagedAttention allows for efficient memory management by:
1. Partitioning attention keys/values into fixed-size blocks
2. Managing these blocks in a KV cache with flexible allocation
3. Reusing blocks across different sequences
4. Avoiding memory fragmentation

Key concepts:
- Physical blocks: Fixed-size memory blocks storing KV cache
- Logical blocks: Virtual blocks that map to physical blocks
- Block tables: Mapping from logical to physical block indices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class BlockConfig:
    """Configuration for KV cache blocks"""
    block_size: int = 16  # Number of tokens per block
    num_heads: int = 8   # Number of attention heads
    head_dim: int = 64   # Dimension per head
    num_blocks: int = 1000  # Total number of physical blocks


class PhysicalBlock:
    """Physical block storing KV cache for a sequence segment"""
    
    def __init__(self, block_size: int, num_heads: int, head_dim: int):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize KV cache: [num_heads, block_size, head_dim]
        self.key_cache = torch.zeros(num_heads, block_size, head_dim)
        self.value_cache = torch.zeros(num_heads, block_size, head_dim)
        self.token_count = 0
        self.allocated = False
    
    def append_kv(self, keys: torch.Tensor, values: torch.Tensor):
        """Append new keys and values to the block"""
        assert self.token_count + keys.shape[1] <= self.block_size
        assert keys.shape[0] == self.num_heads and values.shape[0] == self.num_heads
        assert keys.shape[2] == self.head_dim and values.shape[2] == self.head_dim
        
        start_idx = self.token_count
        end_idx = start_idx + keys.shape[1]
        
        self.key_cache[:, start_idx:end_idx] = keys
        self.value_cache[:, start_idx:end_idx] = values
        self.token_count = end_idx
    
    def get_kv(self, start_idx: int = 0, end_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get keys and values from the block"""
        if end_idx is None:
            end_idx = self.token_count
        return self.key_cache[:, start_idx:end_idx], self.value_cache[:, start_idx:end_idx]


class BlockAllocator:
    """Manages allocation and deallocation of physical blocks"""
    
    def __init__(self, config: BlockConfig):
        self.config = config
        self.blocks = [PhysicalBlock(config.block_size, config.num_heads, config.head_dim) 
                      for _ in range(config.num_blocks)]
        self.free_blocks = set(range(config.num_blocks))
        self.allocated_blocks = {}  # sequence_id -> list of block indices
    
    def allocate(self, sequence_id: int, num_blocks: int) -> List[int]:
        """Allocate blocks for a sequence"""
        if len(self.free_blocks) < num_blocks:
            raise MemoryError(f"Not enough free blocks. Need {num_blocks}, have {len(self.free_blocks)}")
        
        allocated = []
        for _ in range(num_blocks):
            block_idx = self.free_blocks.pop()
            self.blocks[block_idx].allocated = True
            allocated.append(block_idx)
        
        self.allocated_blocks[sequence_id] = allocated
        return allocated
    
    def deallocate(self, sequence_id: int):
        """Deallocate blocks for a sequence"""
        if sequence_id not in self.allocated_blocks:
            return
        
        for block_idx in self.allocated_blocks[sequence_id]:
            self.blocks[block_idx].allocated = False
            self.blocks[block_idx].token_count = 0
            self.free_blocks.add(block_idx)
        
        del self.allocated_blocks[sequence_id]
    
    def get_block(self, block_idx: int) -> PhysicalBlock:
        """Get a physical block by index"""
        return self.blocks[block_idx]


class PagedAttention(nn.Module):
    """PagedAttention mechanism with block-based KV cache management"""
    
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.config = config
        self.allocator = BlockAllocator(config)
        self.block_tables = {}  # sequence_id -> list of physical block indices
        self.sequence_lengths = {}  # sequence_id -> current length
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.num_heads * config.head_dim, config.num_heads * config.head_dim)
        self.k_proj = nn.Linear(config.num_heads * config.head_dim, config.num_heads * config.head_dim)
        self.v_proj = nn.Linear(config.num_heads * config.head_dim, config.num_heads * config.head_dim)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.num_heads * config.head_dim)
    
    def _get_num_blocks_needed(self, seq_len: int) -> int:
        """Calculate number of blocks needed for a sequence length"""
        return (seq_len + self.config.block_size - 1) // self.config.block_size
    
    def allocate_sequence(self, sequence_id: int, max_length: int):
        """Allocate blocks for a new sequence"""
        num_blocks = self._get_num_blocks_needed(max_length)
        block_indices = self.allocator.allocate(sequence_id, num_blocks)
        self.block_tables[sequence_id] = block_indices
        self.sequence_lengths[sequence_id] = 0
    
    def append_kv_cache(self, sequence_id: int, hidden_states: torch.Tensor):
        """Append new tokens to KV cache"""
        if sequence_id not in self.block_tables:
            raise ValueError(f"Sequence {sequence_id} not allocated")
        
        # Project to K, V
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        
        # Reshape: [batch, num_heads, seq_len, head_dim]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        keys = keys.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        
        # Get current sequence info
        current_len = self.sequence_lengths[sequence_id]
        block_indices = self.block_tables[sequence_id]
        
        # Append to appropriate blocks
        tokens_processed = 0
        for i, block_idx in enumerate(block_indices):
            block = self.allocator.get_block(block_idx)
            
            # Calculate how many tokens to add to this block
            block_start = i * self.config.block_size
            block_end = min(block_start + self.config.block_size, current_len + seq_len)
            
            if current_len < block_end:  # This block needs new tokens
                tokens_in_block = min(block_end - current_len, seq_len - tokens_processed)
                
                # Extract keys/values for this block
                start_token = tokens_processed
                end_token = start_token + tokens_in_block
                
                block_keys = keys[0, :, start_token:end_token, :]
                block_values = values[0, :, start_token:end_token, :]
                
                # Calculate offset within the block
                block_offset = current_len - block_start if current_len > block_start else 0
                
                # Append to block (handling offset)
                if block_offset == 0:
                    block.append_kv(block_keys, block_values)
                else:
                    # Handle partial block fill
                    remaining_space = self.config.block_size - block_offset
                    tokens_to_add = min(tokens_in_block, remaining_space)
                    block.append_kv(
                        block_keys[:, :tokens_to_add, :],
                        block_values[:, :tokens_to_add, :]
                    )
                
                tokens_processed += tokens_in_block
            
            if tokens_processed >= seq_len:
                break
        
        self.sequence_lengths[sequence_id] += seq_len
    
    def forward(self, sequence_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with paged attention"""
        if sequence_id not in self.block_tables:
            raise ValueError(f"Sequence {sequence_id} not allocated")
        
        # Project to Q
        batch_size, seq_len, hidden_dim = hidden_states.shape
        query = self.q_proj(hidden_states)
        query = query.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        
        # Gather all keys and values from blocks
        all_keys = []
        all_values = []
        
        block_indices = self.block_tables[sequence_id]
        current_len = self.sequence_lengths[sequence_id]
        
        for i, block_idx in enumerate(block_indices):
            block = self.allocator.get_block(block_idx)
            
            # Calculate actual tokens in this block
            block_start = i * self.config.block_size
            block_end = min(block_start + self.config.block_size, current_len)
            
            if block_start < current_len:  # This block has tokens
                keys, values = block.get_kv(0, block_end - block_start)
                all_keys.append(keys)
                all_values.append(values)
        
        # Concatenate all KV cache
        if all_keys:
            past_keys = torch.cat(all_keys, dim=1)  # [num_heads, total_len, head_dim]
            past_values = torch.cat(all_values, dim=1)
        else:
            past_keys = torch.empty(self.config.num_heads, 0, self.config.head_dim)
            past_values = torch.empty(self.config.num_heads, 0, self.config.head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, past_keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, past_values)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
        output = self.o_proj(attn_output)
        
        return output
    
    def deallocate_sequence(self, sequence_id: int):
        """Deallocate blocks for a sequence"""
        self.allocator.deallocate(sequence_id)
        if sequence_id in self.block_tables:
            del self.block_tables[sequence_id]
        if sequence_id in self.sequence_lengths:
            del self.sequence_lengths[sequence_id]


def demonstrate_paged_attention():
    """Demonstrate PagedAttention functionality"""
    print("=== PagedAttention Demonstration ===\n")
    
    # Configuration
    config = BlockConfig(
        block_size=4,      # Small blocks for demo
        num_heads=4,
        head_dim=32,
        num_blocks=10
    )
    
    # Create PagedAttention layer
    paged_attn = PagedAttention(config)
    
    print(f"Configuration:")
    print(f"  Block size: {config.block_size} tokens")
    print(f"  Number of heads: {config.num_heads}")
    print(f"  Head dimension: {config.head_dim}")
    print(f"  Total blocks: {config.num_blocks}")
    print(f"  Available blocks: {len(paged_attn.allocator.free_blocks)}")
    print()
    
    # Allocate a sequence
    sequence_id = 1
    max_length = 12  # Will need 3 blocks (12 / 4 = 3)
    paged_attn.allocate_sequence(sequence_id, max_length)
    
    print(f"Allocated sequence {sequence_id} with max length {max_length}")
    print(f"Blocks allocated: {paged_attn.block_tables[sequence_id]}")
    print(f"Available blocks: {len(paged_attn.allocator.free_blocks)}")
    print()
    
    # Simulate token processing
    batch_size = 1
    hidden_dim = config.num_heads * config.head_dim
    
    for step in range(3):
        # Generate some tokens (2 tokens per step)
        seq_len = 2
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        print(f"Step {step + 1}: Processing {seq_len} tokens")
        print(f"  Sequence length before: {paged_attn.sequence_lengths[sequence_id]}")
        
        # Append to KV cache
        paged_attn.append_kv_cache(sequence_id, hidden_states)
        
        print(f"  Sequence length after: {paged_attn.sequence_lengths[sequence_id]}")
        
        # Forward pass
        output = paged_attn.forward(sequence_id, hidden_states)
        print(f"  Output shape: {output.shape}")
        
        # Show block utilization
        print(f"  Block utilization:")
        for i, block_idx in enumerate(paged_attn.block_tables[sequence_id]):
            block = paged_attn.allocator.get_block(block_idx)
            print(f"    Block {block_idx}: {block.token_count}/{config.block_size} tokens")
        print()
    
    # Show memory efficiency
    print("Memory Efficiency Analysis:")
    total_allocated = sum(len(blocks) for blocks in paged_attn.block_tables.values())
    total_available = config.num_blocks
    utilization = total_allocated / total_available * 100
    
    print(f"  Allocated blocks: {total_allocated}/{total_available}")
    print(f"  Memory utilization: {utilization:.1f}%")
    print(f"  Total tokens stored: {paged_attn.sequence_lengths[sequence_id]}")
    
    # Demonstrate block reuse
    print("\nDemonstrating block reuse:")
    sequence_id_2 = 2
    max_length_2 = 8  # Will need 2 blocks
    
    try:
        paged_attn.allocate_sequence(sequence_id_2, max_length_2)
        print(f"Successfully allocated sequence {sequence_id_2}")
        print(f"Blocks allocated: {paged_attn.block_tables[sequence_id_2]}")
        print(f"Available blocks: {len(paged_attn.allocator.free_blocks)}")
    except MemoryError as e:
        print(f"Memory error: {e}")
    
    # Cleanup
    paged_attn.deallocate_sequence(sequence_id)
    if sequence_id_2 in paged_attn.block_tables:
        paged_attn.deallocate_sequence(sequence_id_2)
    
    print(f"\nAfter cleanup - Available blocks: {len(paged_attn.allocator.free_blocks)}")


if __name__ == "__main__":
    demonstrate_paged_attention()
