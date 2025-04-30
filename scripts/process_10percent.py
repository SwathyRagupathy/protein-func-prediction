# process_10percent.py

import os
import pandas as pd
from parse_structures import pdb_to_graph, pdb_to_voxel
import torch
from tqdm import tqdm

def process_10percent_pdbs(pdb_folder, graph_output_dir, voxel_output_dir, index_csv_path):
    os.makedirs(graph_output_dir, exist_ok=True)
    os.makedirs(voxel_output_dir, exist_ok=True)

    index_data = []

    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    total_pdbs = len(pdb_files)
    ten_percent_count = int(total_pdbs * 0.10)

    # Select first 10% files
    pdb_files = pdb_files[:ten_percent_count]
    print(f"[INFO] Total PDB files found: {total_pdbs}")
    print(f"[INFO] Processing 10% subset: {len(pdb_files)} files")

    for pdb_file in tqdm(pdb_files, desc="Processing 10% Subset PDB files"):
        try:
            protein_id = os.path.splitext(pdb_file)[0]
            pdb_path = os.path.join(pdb_folder, pdb_file)

            # Graph
            graph = pdb_to_graph(pdb_path)
            graph_path = os.path.join(graph_output_dir, f"graph_{protein_id}.pt")
            torch.save(graph, graph_path)

            # Voxel
            voxel = pdb_to_voxel(pdb_path)
            voxel_path = os.path.join(voxel_output_dir, f"voxel_{protein_id}.pt")
            torch.save(voxel, voxel_path)

            # Index entry
            index_data.append({
                "protein_id": protein_id,
                "graph_path": graph_path,
                "voxel_path": voxel_path
            })

        except Exception as e:
            print(f"[WARNING] Failed to process {pdb_file}: {e}")

    # Save index CSV
    df_index = pd.DataFrame(index_data)
    df_index.to_csv(index_csv_path, index=False)
    print(f"[INFO] Subset processing complete. Index saved to: {index_csv_path}")


# ------------------ Usage ------------------
if __name__ == "__main__":
    pdb_folder = "mane_data"
    graph_output_dir = "ten_percent_graphs"
    voxel_output_dir = "ten_percent_voxels"
    index_csv_path = "ten_percent_protein_index.csv"

    process_10percent_pdbs(pdb_folder, graph_output_dir, voxel_output_dir, index_csv_path)