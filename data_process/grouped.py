import os
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import csv

root_dir = 'D:\GNN_SOURCE\cgcnn\data\porphyrin'
cif_files = [f for f in os.listdir(root_dir) if f.endswith('.cif')]
grouped_spacegroups = {}
for cif_id in cif_files:
    crystal = Structure.from_file(os.path.join(root_dir, cif_id))
    analyzer = SpacegroupAnalyzer(crystal)
    spacegroup = analyzer.get_space_group_symbol()
    obj = cif_id
    if obj in grouped_spacegroups:
        grouped_spacegroups[obj].append(spacegroup)
    else:
        grouped_spacegroups[obj] = [spacegroup]

csv_path = os.path.join('../data', 'grouped', 'grouped_spacegroups.csv')

# 将grouped_spacegroups保存为csv文件
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # 写入数据
    for obj, spacegroups in grouped_spacegroups.items():
        for spacegroup in spacegroups:
            writer.writerow([obj, spacegroup])