import torch
from src.task_vectors import TaskVector

# ============================================================================
# LOAD CHECKPOINTS
# ============================================================================

zs_path_imbalanced = '/scratch/project_2016790/checkpoints/ViT-B-32/imbalanced/BIRADSImbalanced/zeroshot.pt'
ft_path_imbalanced = '/scratch/project_2016790/checkpoints/ViT-B-32/imbalanced/BIRADSImbalanced/finetuned.pt'

zs_path_birads_2 = '/scratch/project_2016790/checkpoints/ViT-B-32/birads-2-focus/BIRADSDensity2Focus/zeroshot.pt'
ft_path_birads_2 = '/scratch/project_2016790/checkpoints/ViT-B-32/birads-2-focus/BIRADSDensity2Focus/finetuned.pt'

zs_path_birads_5 = '/scratch/project_2016790/checkpoints/ViT-B-32/birads-5-focus/BIRADSDensity5Focus/zeroshot.pt'
ft_path_birads_5 = '/scratch/project_2016790/checkpoints/ViT-B-32/birads-5-focus/BIRADSDensity5Focus/finetuned.pt'


# ============================================================================
# CREATE TASK VECTORS
# ============================================================================

print("\n" + "="*70)
print("CREATING TASK VECTORS")
print("="*70)

# Task vector = fine-tuned weights - pretrained weights
task_vector_imbalanced = TaskVector(zs_path_imbalanced, ft_path_imbalanced)
print("✓ Imbalanced task vector created")

task_vector_birads_2 = TaskVector(zs_path_birads_2, ft_path_birads_2)
print("✓ BIRADS 2 focus task vector created")

task_vector_birads_5 = TaskVector(zs_path_birads_5, ft_path_birads_5)
print("✓ BIRADS 5 focus task vector created")


print("\n" + "="*70)
print("CREATING TASK VECTORS")
print("="*70)

task_vector_imbalanced = TaskVector(zs_path_imbalanced, ft_path_imbalanced)
print("✓ Imbalanced task vector created")

task_vector_birads_2 = TaskVector(zs_path_birads_2, ft_path_birads_2)
print("✓ BIRADS 2 focus task vector created")

task_vector_birads_5 = TaskVector(zs_path_birads_5, ft_path_birads_5)
print("✓ BIRADS 5 focus task vector created")


# ============================================================================
# INSPECT TASK VECTOR STRUCTURE
# ============================================================================

print("\n" + "="*70)
print("TASK VECTOR INSPECTION")
print("="*70)

# Task vectors are stored in a .vector dictionary
print(f"\nTask vector is a dictionary with {len(task_vector_imbalanced.vector)} keys")
print("\nAll layer names:")
for i, key in enumerate(task_vector_imbalanced.vector.keys()):
    print(f"  {i+1}. {key}")


# ============================================================================
# COMPARE SHAPES ACROSS ALL THREE TASK VECTORS
# ============================================================================

print("\n" + "="*70)
print("SHAPE COMPARISON")
print("="*70)
print(f"{'Layer Name':<50} {'Imbalanced':<20} {'BIRADS 2':<20} {'BIRADS 5':<20} {'Match?':<10}")
print("-" * 120)

all_match = True
for key in task_vector_imbalanced.vector.keys():
    shape_imb = tuple(task_vector_imbalanced.vector[key].shape)
    shape_b2 = tuple(task_vector_birads_2.vector[key].shape)
    shape_b5 = tuple(task_vector_birads_5.vector[key].shape)
    
    match = (shape_imb == shape_b2 == shape_b5)
    match_str = "✓" if match else "✗"
    
    if not match:
        all_match = False
    
    print(f"{key:<50} {str(shape_imb):<20} {str(shape_b2):<20} {str(shape_b5):<20} {match_str:<10}")

print("-" * 120)
if all_match:
    print("✓ All shapes match across task vectors!")
else:
    print("✗ WARNING: Shape mismatch detected!")


# ============================================================================
# INSPECT FIRST FEW LAYERS IN DETAIL
# ============================================================================

print("\n" + "="*70)
print("DETAILED INSPECTION: First 3 Layers")
print("="*70)

layer_keys = list(task_vector_imbalanced.vector.keys())[:3]

for layer_name in layer_keys:
    print(f"\n{'='*70}")
    print(f"Layer: {layer_name}")
    print(f"{'='*70}")
    
    # Get tensors
    tensor_imb = task_vector_imbalanced.vector[layer_name]
    tensor_b2 = task_vector_birads_2.vector[layer_name]
    tensor_b5 = task_vector_birads_5.vector[layer_name]
    
    print(f"Shape: {tensor_imb.shape}")
    print(f"Dtype: {tensor_imb.dtype}")
    print(f"Device: {tensor_imb.device}")
    
    # Flatten and show first 20 values
    flat_imb = tensor_imb.flatten()
    flat_b2 = tensor_b2.flatten()
    flat_b5 = tensor_b5.flatten()
    
    print(f"\nFirst 20 values:")
    print(f"  Imbalanced:  {flat_imb[:20].tolist()}")
    print(f"  BIRADS 2:    {flat_b2[:20].tolist()}")
    print(f"  BIRADS 5:    {flat_b5[:20].tolist()}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Imbalanced  - Mean: {tensor_imb.mean():.6f}, Std: {tensor_imb.std():.6f}, "
          f"Min: {tensor_imb.min():.6f}, Max: {tensor_imb.max():.6f}")
    print(f"  BIRADS 2    - Mean: {tensor_b2.mean():.6f}, Std: {tensor_b2.std():.6f}, "
          f"Min: {tensor_b2.min():.6f}, Max: {tensor_b2.max():.6f}")
    print(f"  BIRADS 5    - Mean: {tensor_b5.mean():.6f}, Std: {tensor_b5.std():.6f}, "
          f"Min: {tensor_b5.min():.6f}, Max: {tensor_b5.max():.6f}")


# ============================================================================
# COMPUTE ENHANCEMENT VECTOR AND INSPECT
# ============================================================================

print("\n" + "="*70)
print("ENHANCEMENT VECTOR (BIRADS 5 - BIRADS 2)")
print("="*70)

enhancement_vector = task_vector_birads_5 - task_vector_birads_2

print(f"\nEnhancement vector has {len(enhancement_vector.vector)} layers")

# Inspect first layer of enhancement
first_key = layer_keys[0]
enhancement_tensor = enhancement_vector.vector[first_key]

print(f"\nFirst layer: {first_key}")
print(f"Shape: {enhancement_tensor.shape}")

flat_enhancement = enhancement_tensor.flatten()
print(f"\nFirst 20 values of enhancement:")
print(f"  {flat_enhancement[:20].tolist()}")

print(f"\nStatistics:")
print(f"  Mean: {enhancement_tensor.mean():.6f}")
print(f"  Std: {enhancement_tensor.std():.6f}")
print(f"  Min: {enhancement_tensor.min():.6f}")
print(f"  Max: {enhancement_tensor.max():.6f}")


# ============================================================================
# CHECK ENCODER VS CLASSIFICATION HEAD
# ============================================================================

print("\n" + "="*70)
print("ENCODER VS CLASSIFICATION HEAD")
print("="*70)

encoder_keys = [k for k in task_vector_imbalanced.vector.keys() if 'classification_head' not in k]
head_keys = [k for k in task_vector_imbalanced.vector.keys() if 'classification_head' in k]

print(f"\nEncoder layers: {len(encoder_keys)}")
print(f"Classification head layers: {len(head_keys)}")

if head_keys:
    print(f"\nClassification head layers:")
    for key in head_keys:
        shape = task_vector_imbalanced.vector[key].shape
        print(f"  {key}: {shape}")
        
        # Show values
        tensor = task_vector_imbalanced.vector[key]
        flat = tensor.flatten()
        print(f"    First 20 values: {flat[:20].tolist()}")
else:
    print("\n✓ No classification head in task vectors (frozen during training)")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("INSPECTION SUMMARY")
print("="*70)
print(f"✓ Task vectors created successfully")
print(f"✓ All shapes match: {all_match}")
print(f"✓ Total layers: {len(task_vector_imbalanced.vector)}")
print(f"✓ Encoder layers: {len(encoder_keys)}")
print(f"✓ Classification head layers: {len(head_keys)}")
print("="*70)

# ============================================================================
# APPLY TASK ANALOGY WITH DIFFERENT ALPHAS
# ============================================================================

print("\n" + "="*70)
print("APPLYING TASK ANALOGY")
print("="*70)

# Test different scaling factors
alphas = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]

for alpha in alphas:
    print(f"\n--- Alpha = {alpha} ---")
    
    # Apply analogy: imbalanced + α * enhancement
    enhanced_task_vector = task_vector_imbalanced + (alpha * enhancement_vector)
    
    # Apply enhanced vector to zero-shot model
    enhanced_model_path = f'/scratch/project_2016790/checkpoints/ViT-B-32/enhanced_alpha_{alpha}.pt'
    enhanced_model = enhanced_task_vector.apply_to(zs_path_imbalanced, scaling_coef=1.0)
    
    # Save enhanced model
    torch.save(enhanced_model, enhanced_model_path)
    print(f"✓ Saved enhanced model to: {enhanced_model_path}")


# ============================================================================
# SAVE ALL TASK VECTORS FOR LATER ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("SAVING TASK VECTORS")
print("="*70)

task_vectors_save_path = '/scratch/project_2016790/checkpoints/ViT-B-32/all_task_vectors.pt'

torch.save({
    'imbalanced': task_vector_imbalanced,
    'birads_2_focus': task_vector_birads_2,
    'birads_5_focus': task_vector_birads_5,
    'enhancement': enhancement_vector,
}, task_vectors_save_path)

print(f"✓ All task vectors saved to: {task_vectors_save_path}")
print("="*70)


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TASK VECTOR CREATION COMPLETE")
print("="*70)
print(f"Task vectors created:")
print(f"  1. Imbalanced baseline")
print(f"  2. BIRADS 2 focus")
print(f"  3. BIRADS 5 focus")
print(f"  4. Enhancement vector")
print(f"\nEnhanced models created for alphas: {alphas}")
print(f"\nNext step: Evaluate enhanced models on the imbalanced test set")
print("="*70)
