import os
import torch
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from torch_geometric.explain import GNNExplainer

def run_explainability(hybrid_model=None, test_loader=None, device=None, save_prefix="exai"):
    os.makedirs(save_prefix, exist_ok=True)
    model.eval()

    for i, (graph, voxel, labels) in enumerate(dataloader):
        try:
            graph = graph.to(device)
            voxel = voxel.to(device)
            labels = labels.to(device)

            # ----------- GNNExplainer -------------
            explainer = GNNExplainer(hybrid_model.gnn)
            _, edge_mask = explainer.explain_graph(graph.x, graph.edge_index)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"Sample {i} - GNN Edge Importance")
            ax.bar(range(len(edge_mask)), edge_mask.detach().cpu().numpy())
            plt.tight_layout()
            plt.savefig(os.path.join(save_prefix, f"gnn_edge_importance_{i}.png"))
            plt.close()

            # ----------- Grad-CAM (CNN) -------------
            gradcam = LayerGradCam(model.cnn, model.cnn.conv2)
            voxel.requires_grad_()
            output = model(graph, voxel)
            class_idx = output[0].argmax().item()

            cam = gradcam.attribute(voxel, target=class_idx)
            slice_index = voxel.shape[2] // 2
            cam_np = cam[0, 0, slice_index].detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cam_np, cmap="inferno")
            ax.set_title(f"Sample {i} - GradCAM Slice | Class {class_idx}")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_prefix, f"gradcam_slice_{i}.png"))
            plt.close()

        except Exception as e:
            print(f"[⚠️ Warning] Skipping sample {i} due to: {e}")