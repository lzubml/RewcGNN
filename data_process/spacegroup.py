import os
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import csv

#这个文件用于生成所有cif对应的空间群名称
# 用您的cif文件夹路径替换'your_cif_folder'
cif_folder = 'D:\GNN_SOURCE\cgcnn\data\porphyrin'
cif_files = [f for f in os.listdir(cif_folder) if f.endswith('.cif')]
# 遍历cif文件
#定义空间群集合
spacegroups = []
for cif_file in cif_files:
    # 从cif文件中提取结构信息
    structure = Structure.from_file(os.path.join(cif_folder, cif_file))
    #获取晶体空间群
    analyzer = SpacegroupAnalyzer(structure)
    spacegroup = analyzer.get_space_group_symbol()
    spacegroups.append(spacegroup)
#共41种空间群
unique_spacegroups = set(spacegroups)
spacegroups_list = list(unique_spacegroups)

with open('spacegroups.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(spacegroups_list)

