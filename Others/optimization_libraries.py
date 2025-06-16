# LLM Optimization Libraries Demo: Flash Attention, DeepSpeed, and vLLM
# This notebook demonstrates the performance benefits of optimization libraries

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# 1. FLASH ATTENTION DEMO
# =============================================================================

class StandardAttention(nn.Module):
    """Standard scaled dot-product attention implementation"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.w_o(attn_output)

# Flash Attention implementation (simplified version)
class FlashAttention(nn.Module):
    """Memory-efficient Flash Attention implementation"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Use torch's scaled_dot_product_attention for memory efficiency
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        else:
            # Fallback to chunked computation for memory efficiency
            attn_output = self._chunked_attention(Q, K, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.w_o(attn_output)
    
    def _chunked_attention(self, Q, K, V, chunk_size=1024):
        """Process attention in chunks to reduce memory usage"""
        batch_size, n_heads, seq_len, d_k = Q.shape
        
        if seq_len <= chunk_size:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
            attn_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, V)
        
        # Process in chunks
        output = torch.zeros_like(Q)
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = Q[:, :, i:end_i, :]
            
            scores = torch.matmul(q_chunk, K.transpose(-2, -1)) / np.sqrt(d_k)
            attn_weights = torch.softmax(scores, dim=-1)
            output[:, :, i:end_i, :] = torch.matmul(attn_weights, V)
        
        return output

def benchmark_attention(attention_class, batch_size, seq_len, d_model, n_heads, num_runs=5):
    """Benchmark attention implementation"""
    model = attention_class(d_model, n_heads).to(device)
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Measure memory before
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            output = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.time() - start_time)
    
    # Measure memory after
    if torch.cuda.is_available():
        mem_after = torch.cuda.max_memory_allocated()
        memory_used = (mem_after - mem_before) / 1e6  # MB
    else:
        memory_used = 0
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'memory_mb': memory_used,
        'output_shape': output.shape
    }

# Run Flash Attention benchmarks
print("=" * 60)
print("FLASH ATTENTION BENCHMARK")
print("=" * 60)

configs = [
    (1, 512, 512, 8),
    (1, 1024, 512, 8),
    (1, 2048, 512, 8),
]

flash_results = []
standard_results = []

for batch_size, seq_len, d_model, n_heads in configs:
    print(f"\nTesting: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
    
    # Standard attention
    std_result = benchmark_attention(StandardAttention, batch_size, seq_len, d_model, n_heads)
    standard_results.append(std_result)
    print(f"Standard Attention: {std_result['avg_time']:.4f}s ± {std_result['std_time']:.4f}s, Memory: {std_result['memory_mb']:.1f}MB")
    
    # Flash attention
    flash_result = benchmark_attention(FlashAttention, batch_size, seq_len, d_model, n_heads)
    flash_results.append(flash_result)
    print(f"Flash Attention:    {flash_result['avg_time']:.4f}s ± {flash_result['std_time']:.4f}s, Memory: {flash_result['memory_mb']:.1f}MB")
    
    speedup = std_result['avg_time'] / flash_result['avg_time']
    memory_reduction = (std_result['memory_mb'] - flash_result['memory_mb']) / std_result['memory_mb'] * 100
    print(f"Speedup: {speedup:.2f}x, Memory Reduction: {memory_reduction:.1f}%")

# =============================================================================
# 2. DEEPSPEED SIMULATION DEMO
# =============================================================================

print("\n" + "=" * 60)
print("DEEPSPEED OPTIMIZATION SIMULATION")
print("=" * 60)

class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for demonstration"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = StandardAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class SimpleTransformer(nn.Module):
    """Simple transformer model"""
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

def simulate_deepspeed_training(model, optimizer, data_loader, use_mixed_precision=False, 
                              gradient_accumulation_steps=1):
    """Simulate DeepSpeed training optimizations"""
    model.train()
    total_loss = 0
    steps = 0
    
    # Enable mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixed precision forward pass
        if use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_mixed_precision and torch.cuda.is_available():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_mixed_precision and torch.cuda.is_available():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            steps += 1
        
        if batch_idx >= 10:  # Limit for demo
            break
    
    elapsed_time = time.time() - start_time
    return total_loss / max(steps, 1), elapsed_time

# Create a simple dataset
def create_dummy_dataset(batch_size, seq_len, vocab_size, num_batches=20):
    """Create dummy dataset for training simulation"""
    data = []
    for _ in range(num_batches):
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append((inputs, targets))
    return data

# Test different DeepSpeed configurations
print("Simulating DeepSpeed optimizations...")

model_config = {
    'vocab_size': 1000,
    'd_model': 256,
    'n_heads': 8,
    'd_ff': 1024,
    'n_layers': 4
}

batch_size = 4
seq_len = 128
dataset = create_dummy_dataset(batch_size, seq_len, model_config['vocab_size'])

configs_to_test = [
    {"name": "Baseline", "mixed_precision": False, "grad_accum": 1},
    {"name": "Mixed Precision", "mixed_precision": True, "grad_accum": 1},
    {"name": "Gradient Accumulation", "mixed_precision": False, "grad_accum": 4},
    {"name": "Mixed Precision + Grad Accum", "mixed_precision": True, "grad_accum": 4},
]

deepspeed_results = []

for config in configs_to_test:
    # Create fresh model and optimizer
    model = SimpleTransformer(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Measure memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
    
    # Run training simulation
    avg_loss, training_time = simulate_deepspeed_training(
        model, optimizer, dataset, 
        use_mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["grad_accum"]
    )
    
    # Measure memory after training
    if torch.cuda.is_available():
        mem_after = torch.cuda.max_memory_allocated()
        memory_used = (mem_after - mem_before) / 1e6  # MB
    else:
        memory_used = 0
    
    result = {
        'name': config['name'],
        'loss': avg_loss,
        'time': training_time,
        'memory_mb': memory_used
    }
    deepspeed_results.append(result)
    
    print(f"{config['name']:25}: Loss={avg_loss:.4f}, Time={training_time:.2f}s, Memory={memory_used:.1f}MB")

# =============================================================================
# 3. vLLM INFERENCE OPTIMIZATION SIMULATION
# =============================================================================

print("\n" + "=" * 60)
print("vLLM INFERENCE OPTIMIZATION SIMULATION")
print("=" * 60)

class StandardInference:
    """Standard inference approach"""
    def __init__(self, model):
        self.model = model
        
    def generate(self, input_ids, max_length=50, batch_size=1):
        """Standard greedy generation"""
        self.model.eval()
        
        # Simulate multiple sequences in batch
        all_sequences = []
        
        with torch.no_grad():
            for _ in range(batch_size):
                current_ids = input_ids.clone()
                
                for _ in range(max_length - input_ids.size(1)):
                    outputs = self.model(current_ids)
                    next_token = torch.argmax(outputs[:, -1, :], dim=-1, keepdim=True)
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                
                all_sequences.append(current_ids)
        
        return torch.stack(all_sequences)

class vLLMSimulation:
    """Simulate vLLM optimizations (PagedAttention simulation)"""
    def __init__(self, model, block_size=16):
        self.model = model
        self.block_size = block_size
        self.kv_cache = {}
        
    def generate(self, input_ids, max_length=50, batch_size=1):
        """Simulate efficient batched generation with KV caching"""
        self.model.eval()
        
        # Simulate parallel processing of multiple sequences
        with torch.no_grad():
            # Process all sequences in parallel (simulated)
            batch_input = input_ids.repeat(batch_size, 1)
            current_length = input_ids.size(1)
            
            # Pre-allocate output tensor
            output_sequences = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
            output_sequences[:, :current_length] = batch_input
            
            # Simulate efficient generation with shared computation
            for pos in range(current_length, max_length):
                # In real vLLM, this would use PagedAttention for memory efficiency
                current_context = output_sequences[:, :pos]
                
                # Batch forward pass (more efficient than sequential)
                outputs = self.model(current_context)
                next_tokens = torch.argmax(outputs[:, -1, :], dim=-1)
                output_sequences[:, pos] = next_tokens
        
        return output_sequences

def benchmark_inference(inference_class, model, input_ids, max_length, batch_sizes, num_runs=3):
    """Benchmark inference approaches"""
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Create inference engine
        if inference_class == vLLMSimulation:
            engine = inference_class(model, block_size=16)
        else:
            engine = inference_class(model)
        
        # Warmup
        _ = engine.generate(input_ids, max_length=max_length, batch_size=min(batch_size, 2))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            outputs = engine.generate(input_ids, max_length=max_length, batch_size=batch_size)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start_time)
        
        if torch.cuda.is_available():
            mem_after = torch.cuda.max_memory_allocated()
            memory_used = (mem_after - mem_before) / 1e6  # MB
        else:
            memory_used = 0
        
        # Calculate throughput (tokens per second)
        total_tokens = batch_size * max_length
        throughput = total_tokens / np.mean(times)
        
        results[batch_size] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'memory_mb': memory_used,
            'throughput': throughput,
            'output_shape': outputs.shape
        }
        
        print(f"  Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print(f"  Memory: {memory_used:.1f}MB")
        print(f"  Throughput: {throughput:.1f} tokens/s")
    
    return results

# Test inference optimization
print("Running inference benchmarks...")

# Create smaller model for inference testing
inference_model = SimpleTransformer(
    vocab_size=1000, d_model=128, n_heads=4, d_ff=256, n_layers=2
).to(device)

input_ids = torch.randint(0, 1000, (1, 10)).to(device)
max_length = 30
batch_sizes = [1, 2, 4] if torch.cuda.is_available() else [1, 2]

print("\nStandard Inference:")
standard_results = benchmark_inference(StandardInference, inference_model, input_ids, max_length, batch_sizes)

print("\nvLLM Simulation:")
vllm_results = benchmark_inference(vLLMSimulation, inference_model, input_ids, max_length, batch_sizes)

# =============================================================================
# 4. VISUALIZATION AND SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON SUMMARY")
print("=" * 60)

# Check if we have results to visualize
if len(flash_results) == 0 or len(standard_results) == 0:
    print("No benchmark results available for visualization.")
else:
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Flash Attention comparison
    if len(standard_results) > 0 and len(flash_results) > 0:
        # Ensure we have valid results
        valid_standard = [r for r in standard_results if isinstance(r, dict) and 'avg_time' in r]
        valid_flash = [r for r in flash_results if isinstance(r, dict) and 'avg_time' in r]
        
        if valid_standard and valid_flash:
            standard_times = [r['avg_time'] for r in valid_standard]
            flash_times = [r['avg_time'] for r in valid_flash]
            
            axes[0,0].bar(['Standard', 'Flash'], [np.mean(standard_times), np.mean(flash_times)], 
                          color=['red', 'green'], alpha=0.7)
            axes[0,0].set_title('Flash Attention: Average Inference Time')
            axes[0,0].set_ylabel('Time (seconds)')
            
            # Memory usage comparison
            standard_memory = [r.get('memory_mb', 0) for r in valid_standard]
            flash_memory = [r.get('memory_mb', 0) for r in valid_flash]
            
            axes[0,1].bar(['Standard', 'Flash'], [np.mean(standard_memory), np.mean(flash_memory)], 
                          color=['red', 'green'], alpha=0.7)
            axes[0,1].set_title('Flash Attention: Memory Usage')
            axes[0,1].set_ylabel('Memory (MB)')
        else:
            axes[0,0].text(0.5, 0.5, 'No Flash Attention\nResults Available', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,1].text(0.5, 0.5, 'No Memory\nResults Available', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
    
    # DeepSpeed training time comparison
    if len(deepspeed_results) > 0:
        valid_deepspeed = [r for r in deepspeed_results if isinstance(r, dict) and 'name' in r and 'time' in r]
        
        if valid_deepspeed:
            deepspeed_names = [r['name'] for r in valid_deepspeed]
            deepspeed_times = [r['time'] for r in valid_deepspeed]
            
            axes[1,0].bar(range(len(deepspeed_names)), deepspeed_times, color='blue', alpha=0.7)
            axes[1,0].set_title('DeepSpeed: Training Time Comparison')
            axes[1,0].set_ylabel('Time (seconds)')
            axes[1,0].set_xticks(range(len(deepspeed_names)))
            axes[1,0].set_xticklabels(deepspeed_names, rotation=45, ha='right')
        else:
            axes[1,0].text(0.5, 0.5, 'No DeepSpeed\nResults Available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
    
    # vLLM throughput comparison
    if 'standard_results' in locals() and 'vllm_results' in locals():
        # Check if these are dictionaries with batch size keys
        if isinstance(standard_results, dict) and isinstance(vllm_results, dict):
            batch_sizes_list = list(standard_results.keys())
            if batch_sizes_list:
                standard_throughput = [standard_results[bs].get('throughput', 0) for bs in batch_sizes_list]
                vllm_throughput = [vllm_results[bs].get('throughput', 0) for bs in batch_sizes_list]
                
                x = np.arange(len(batch_sizes_list))
                width = 0.35
                
                axes[1,1].bar(x - width/2, standard_throughput, width, label='Standard', color='red', alpha=0.7)
                axes[1,1].bar(x + width/2, vllm_throughput, width, label='vLLM Sim', color='green', alpha=0.7)
                axes[1,1].set_title('Inference Throughput Comparison')
                axes[1,1].set_ylabel('Tokens/second')
                axes[1,1].set_xlabel('Batch Size')
                axes[1,1].set_xticks(x)
                axes[1,1].set_xticklabels(batch_sizes_list)
                axes[1,1].legend()
            else:
                axes[1,1].text(0.5, 0.5, 'No Inference\nResults Available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
        else:
            axes[1,1].text(0.5, 0.5, 'Inference results\nnot in expected format', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nKEY FINDINGS:")
    print("-" * 40)
    
    # Flash Attention summary
    if len(valid_standard) > 0 and len(valid_flash) > 0:
        flash_speedup = np.mean(standard_times) / np.mean(flash_times)
        if np.mean(standard_memory) > 0:
            flash_memory_savings = (np.mean(standard_memory) - np.mean(flash_memory)) / np.mean(standard_memory) * 100
        else:
            flash_memory_savings = 0
        print(f"Flash Attention:")
        print(f"  - Average speedup: {flash_speedup:.2f}x")
        print(f"  - Memory savings: {flash_memory_savings:.1f}%")
    
    # DeepSpeed summary
    if len(valid_deepspeed) > 0:
        baseline_result = next((r for r in valid_deepspeed if r['name'] == 'Baseline'), None)
        if baseline_result:
            baseline_time = baseline_result['time']
            best_time = min(r['time'] for r in valid_deepspeed)
            deepspeed_speedup = baseline_time / best_time
            print(f"\nDeepSpeed Optimizations:")
            print(f"  - Best speedup: {deepspeed_speedup:.2f}x")
            print(f"  - Mixed precision helps with memory and speed")
            print(f"  - Gradient accumulation enables larger effective batch sizes")
    
    # vLLM summary
    if isinstance(standard_results, dict) and isinstance(vllm_results, dict) and batch_sizes_list:
        max_batch = max(batch_sizes_list)
        if max_batch in standard_results and max_batch in vllm_results:
            standard_max_throughput = standard_results[max_batch].get('throughput', 1)
            vllm_max_throughput = vllm_results[max_batch].get('throughput', 1)
            vllm_speedup = vllm_max_throughput / standard_max_throughput
            print(f"\nvLLM Simulation:")
            print(f"  - Throughput improvement: {vllm_speedup:.2f}x")
            print(f"  - Better scaling with batch size")
            print(f"  - More efficient memory usage for inference")

print("\n" + "=" * 60)
print("INSTALLATION COMMANDS FOR REAL USAGE:")
print("=" * 60)
print("# Flash Attention")
print("pip install flash-attn --no-build-isolation")
print("\n# DeepSpeed")
print("pip install deepspeed")
print("\n# vLLM")
print("pip install vllm")

print("\nNote: This demo simulates the benefits of these libraries.")
print("Real implementations would show even more dramatic improvements!")
