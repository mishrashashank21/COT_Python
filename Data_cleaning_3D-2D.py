import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import math, json, trimesh, pyclipper
from random import random
from matplotlib.patches import Polygon
import shapely.geometry
import shapely.wkt
from matplotlib.collections import PatchCollection
import geopandas as gpd

# path_lab = r'C:\Users\sutd\Dropbox\PycharmProjects\COT_Python\revit_copy.json'


# path_lab = r'C:\Users\Shashank\Dropbox\PycharmProjects\COT_Python\revit_copy.json'
path_lab = r"1505_Block_AS3_copy.json"


# path_lab = '21003_A_HostelZ2_A_R22_copy.json'


def Plot(df_process, level, df_process_2, attributes, color='black', offset=0.0):
    df_process_final_group = df_process.groupby(('Attributes', 'Category.Name'))

    fig, ax = plt.subplots(figsize=(66, 66))
    bounds = [[1e5, 1e5], [-1e5, -1e5]]

    attributes_index = 0

    # images = []

    for key in df_process_final_group.groups.keys():
        print("KEY of CROSS_SECTION  {} ".format(key))
        # points = np.array(geometry['Points'])
        # planes = np.array(geometry['Planes'])
        points = df_process_final_group.get_group(key).loc[:, ('Geometry', 'Points')].values
        planes = df_process_final_group.get_group(key).loc[:, ('Geometry', 'Planes')].values
        # labels = df_process.loc[:, ('Attributes', 'Type')].values
        labels = df_process_final_group.get_group(key).loc[:, ('Attributes', 'Category.Name')].values

        texts = df_process_final_group.get_group(key).loc[:, ('Attributes', 'Name')].values
        # print(points.shape[0])

        # cmap_list = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
        #              'BuPu',
        #              'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        # cmap_list = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter',
        #              'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']

        cmap_list = ['Greys', 'Blues', 'Greens', 'Oranges']

        # attributes_index = np.arrange(0,len(attributes))

        # attributes.remove(labels[0])

        patches = []

        for i in range(points.shape[0]):
            j = np.array(points[i])
            k = np.array(planes[i])

            # print("Key  {}".format(key))

            # print(df_process_final_group.get_group(key).index[i])
            # df_process_2.loc['1b283f42-2e66-493f-8d2e-754ca0dee621-00050a33', [('Geometry', 'Shape')]] = 2
            # print(df_process_2.loc['1b283f42-2e66-493f-8d2e-754ca0dee621-00050a33', [('Geometry', 'Shape')]])

            mesh = trimesh.Trimesh(j, k)
            # mesh.show()
            # print("Mesh")
            # print(mesh)
            origin = mesh.centroid  # + np.array( [0, 0, offset] )
            # print(origin)
            origin1 = mesh.centroid + np.array([0, 0, 1 - origin[2]])
            sliced = mesh.section(plane_origin=origin, plane_normal=[0, 0, 1])
            # sliced.show()
            # print("Slice")
            # print(sliced)
            # if (sliced is None): return
            if (sliced is None):
                print(df_process_final_group.get_group(key).index[i])
                continue
            slice, xform = sliced.to_planar()
            # print(slice.polygons_full)

            all_polygons = slice.polygons_full
            polys = []
            for poly in all_polygons:
                polys.append([poly])

            # print(polys)

            df_loc = df_process_2.loc[df_process_final_group.get_group(key).index[i], [('Geometry', 'Shape')]]
            # print(df_loc.shape)
            # print(df_loc.head)
            #
            # print(df_loc.dtypes)
            # print(df_process_2.dtypes)

            if df_loc.shape[0] == 1:

                df_process_2.loc[df_process_final_group.get_group(key).index[i], [(
                    'Geometry', 'Shape')]] = polys
            else:

                for g in range(0, df_loc.shape[0]):
                    df_loc.iloc[g, 0] = all_polygons

                df_process_2.loc[df_process_final_group.get_group(key).index[i], [('Geometry', 'Shape')]] = df_loc[
                    'Geometry', 'Shape'].values

            # df_process_2.loc[df_process_final_group.get_group(key).index[i], [(
            #     'Geometry', 'Shape')]] = all_polygons

            for polygon in slice.polygons_full:
                points_poly = np.array([[xform[0, 3] + p[0], xform[1, 3] + p[1]]
                                        for p in polygon.exterior.coords])
                # r = int(255 * random())
                # g = int(255 * random())
                # b = int(255 * random())
                # c = f'#{r:02X}{g:02X}{b:02X}'
                # ax.add_patch(Polygon(points_poly, fill=True, facecolor=c, alpha=0.4, label=texts[i]))
                # ax.annotate(text=labels[i], xy=(origin[0], origin[1]), horizontalalignment='center')

                polygon = Polygon(points_poly, fill=True, alpha=0.4)
                patches.append(polygon)

                pmin = np.min(points_poly, axis=0)
                pmax = np.max(points_poly, axis=0)
                bounds[0][0] = min(bounds[0][0], pmin[0])
                bounds[0][1] = min(bounds[0][1], pmin[1])
                bounds[1][0] = max(bounds[1][0], pmax[0])
                bounds[1][1] = max(bounds[1][1], pmax[1])

                # print(df_process.index[i])

        # cmap = plt.get_cmap(cmap_list[attributes_index])
        # attributes.remove(labels[i])

        cmap = plt.get_cmap(cmap_list[attributes_index])

        n_patches = np.arange(0, len(patches) + 1)
        # n_patches = [i for i in range(10, len(patches) + 11)]

        colors = cmap(n_patches)

        collection = PatchCollection(patches)

        collection.set_color(colors)

        ax.add_collection(collection)

        # patches.clear()

        attributes_index = attributes_index + 1

    padding = max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]) * 0.1
    ax.set_xlim(bounds[0][0] - padding, bounds[1][0] + padding)
    ax.set_ylim(bounds[0][1] - padding, bounds[1][1] + padding)
    # ax.set_aspect(1.0)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # def legend_without_duplicate_labels(figure):
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     figure.legend(by_label.values(), by_label.keys(), loc='upper right')
    #
    #
    # legend_without_duplicate_labels(fig)
    # fig.legend(fontsize=30, loc='upper right')

    # for i in range(len(images)):
    #     bar = plt.colorbar(images[i])
    #     bar.set_label('ColorBar ' + str(i))
    # plt.colorbar()
    plt.title(f"Level:{level}", fontsize=50)
    plt.show()
    # plt.savefig("smu.png")
    return df_process_2
    # exit(0)


with open(path_lab) as file:
    data = json.load(file)
d = data.values()

df = pd.read_json(path_lab, orient='index')
# print(df.head())
print("ORIGINAL DATA  {}".format(df.shape[0]))

df_explode = df.explode("Geometry")
index = df_explode.index.tolist()
print("DATA AFTER EXPLODING GEOMETRY  {}".format(len(index)))

df_explode_geometry = pd.json_normalize(df_explode.iloc[:, 0])
header = [["Geometry" for x in range(len(df_explode_geometry.columns))], df_explode_geometry.columns]
df_explode_geometry.columns = header
# df_explode_geometry = [f'{i}{j}' for i, j in df_explode_geometry.columns]
# df_explode_geometry.columns = df_explode_geometry.columns.map(''.join)

df_explode_materials = pd.json_normalize(df_explode.iloc[:, 1])
df_explode_materials = pd.json_normalize(df_explode_materials.iloc[:, 0])
header = [["Materials" for x in range(len(df_explode_materials.columns))], df_explode_materials.columns]
df_explode_materials.columns = header

df_explode_attributes = pd.json_normalize(df_explode.iloc[:, 2])
header = [["Attributes" for x in range(len(df_explode_attributes.columns))], df_explode_attributes.columns]
df_explode_attributes.columns = header

frames = [df_explode_geometry, df_explode_materials, df_explode_attributes]
df_new = pd.concat(frames, axis=1)

df_new.set_axis(index, axis=0, inplace=True)
# print(df_new.head())
# print(df_new.index.name)
df_new.index.rename('Entity', inplace=True)
print("DATA AFTER EXPLODING AND NORMALIZING  {}".format(df_new.shape))

# nan_value = float("NaN")
# df_new[('Geometry', 'Type')].replace("", nan_value, inplace=True)
# df_new[('Geometry', 'Type')].fillna('No Geometry', inplace=True)
# df_new_group_geometry_type = df_new.groupby(('Geometry', 'Type'))
# print(df_new_group_geometry_type.groups.keys())

# print(df_new_group_geometry_type.get_group('Mesh').shape)
# print(df_new_group_geometry_type.get_group('No Geometry').shape)
# print(df_new_group_geometry_type.get_group('Node').shape)
# print(df_new_group_geometry_type.get_group('Poly').shape)
#
# grouped_mesh = df_new_group_geometry_type.get_group('Mesh').groupby(('Attributes', 'Category.Name'))
# grouped_no = df_new_group_geometry_type.get_group('No Geometry').groupby(('Attributes', 'Category.Name'))
# grouped_node = df_new_group_geometry_type.get_group('Node').groupby(('Attributes', 'Category.Name'))
# grouped_poly = df_new_group_geometry_type.get_group('Poly').groupby(('Attributes', 'Category.Name'))
# d = grouped_poly.size()
# # print(grouped_mesh.size())
# print(grouped_no.size())
# print(grouped_poly.size())
# exit(0)

df_new_group = df_new.groupby('Entity')
print("MATCHING LENGTH AFTER EXPLODING AND NORMALIZING  {}".format(
    len(np.unique(np.array(list(df_new_group.groups.keys()))))))

features = df_new.columns.tolist()
df_features = pd.DataFrame(features, columns=['Index1', 'Index2'])
# print(df_new.columns.tolist())

# df_new_group_attributes_type = df_new.groupby(('Attributes', 'Type'))
# df_new_group_attributes_name = df_new.groupby(('Attributes', 'Name'))

df_new_group_attributes_type = df_new.groupby(('Attributes', 'Type'))
# df_new_group_attributes_1 = df_new.groupby([('Attributes', 'Name'), ('Attributes', 'Type')])
df_new_group_attributes_category_name = df_new.groupby(('Attributes', 'Category.Name'))

Att_list_type = list(df_new_group_attributes_type.groups.keys())
Att_list_category_name = list(df_new_group_attributes_category_name.groups.keys())

print(len(Att_list_type))
print(Att_list_type)
print(len(Att_list_category_name))
print(Att_list_category_name)
# exit(0)

# print(df_new_group_attributes_type.groups.keys())
# print(df_new_group_attributes_name.groups.keys())
# print(df_new.loc[:, ('Attributes', 'Detail Level')].unique())

# attributes = ['Floor', 'Room', 'Stairs', 'Wall', 'Doors']
attributes = ['Rooms', 'Walls', 'Doors']
# attributes = ['Floors']

level_name = list(df_new_group_attributes_type.get_group('Level')[('Attributes', 'Name')])
level_guid = list(df_new_group_attributes_type.get_group('Level')[('Attributes', 'Guid')])

df_process = pd.DataFrame()
for i in attributes:
    if i in Att_list_type:
        # print(df_new_group_attributes_type.get_group(i).shape)
        df_process = df_process.append(df_new_group_attributes_type.get_group(i))

    if i in Att_list_category_name:
        # print(df_new_group_attributes_category_name.get_group(i).shape)
        df_process = df_process.append(df_new_group_attributes_category_name.get_group(i))

print("SHAPE OF DATA AFTER FILTER ATTRIBUTES  {}".format(df_process.shape))

df_process[('Attributes', 'Base Level')].fillna(0, inplace=True)
df_process[('Attributes', 'Base Constraint')].fillna(0, inplace=True)

df_process[('Attributes', 'Base Constraint')] = df_process[('Attributes', 'Base Constraint')].astype('int')
df_process[('Attributes', 'Base Level')] = df_process[('Attributes', 'Base Level')].astype('int')

df_process[('Attributes', 'Base Constraint')] = df_process[('Attributes', 'Base Constraint')].round(decimals=0)
df_process[('Attributes', 'Base Level')] = df_process[('Attributes', 'Base Level')].round(decimals=0)

# print(df_process[('Geometry', 'Type')])

df_process_2 = pd.DataFrame()

keys = [('Geometry', 'Type'), ('Geometry', 'Points'), ('Geometry', 'Planes'), ('Attributes', 'Type'),
        ('Attributes', 'Name'), ('Attributes', 'Guid'),
        ('Attributes', 'Category.Name'),
        ('Attributes', 'Level'), ('Attributes', 'Top Constraint'), ('Attributes', 'Base Constraint'),
        ('Attributes', 'Top Level'), ('Attributes', 'Base Level')]

for key in keys:
    df_process_2[key] = df_process[key]

df_process_2[('Geometry', 'Shape')] = ''
df_process_2[('Geometry', 'Shape')] = df_process_2[('Geometry', 'Shape')].astype('object')

# check2 = df_process_2.loc['1b283f42-2e66-493f-8d2e-754ca0dee621-00050a33'
# , [('Geometry', 'Shape')]]

nan_value = float("NaN")
df_process_2[('Geometry', 'Type')].replace("", nan_value, inplace=True)
df_process_2.dropna(subset=[('Geometry', 'Type')], inplace=True)

df_process_2[('Attributes', 'Base Level')].fillna(0, inplace=True)
df_process_2[('Attributes', 'Base Constraint')].fillna(0, inplace=True)

df_process_2[('Attributes', 'Base Constraint')] = df_process_2[('Attributes', 'Base Constraint')].astype('int')
df_process_2[('Attributes', 'Base Level')] = df_process_2[('Attributes', 'Base Level')].astype('int')

print(df_process_2[('Attributes', 'Base Constraint')].dtype)

# decimals = 0
# df_process_2[('Attributes', 'Base Constraint')] = df_process_2[('Attributes', 'Base Constraint')].apply(
#     lambda x: round(x, decimals))
# df_process_2[('Attributes', 'Base Level')] = df_process_2[('Attributes', 'Base Level')].apply(
#     lambda x: round(x, decimals))

df_process_2[('Attributes', 'Base Constraint')] = df_process_2[('Attributes', 'Base Constraint')].round(decimals=0)
df_process_2[('Attributes', 'Base Level')] = df_process_2[('Attributes', 'Base Level')].round(decimals=0)

print(df_process[('Attributes', 'Guid')].dtypes)
print(df_process[('Attributes', 'Level')].dtypes)

nan_value = float("NaN")
df_process[('Geometry', 'Type')].replace("", nan_value, inplace=True)
df_process.dropna(subset=[('Geometry', 'Type')], inplace=True)

print("LENGTH OF DATA AFTER FILTER ATTRIBUTES ADN REMOVING NULLS  {}".format(df_process.shape[0]))

# exit(0)
df_process_group_level = df_process.groupby(('Attributes', 'Level'))
df_process_group_base_constraint = df_process.groupby(('Attributes', 'Base Constraint'))
df_process_group_base_level = df_process.groupby(('Attributes', 'Base Level'))

print(df_process_group_level.groups.keys())
print(df_process_group_base_constraint.groups.keys())
# exit(0)
df_process_final = pd.DataFrame()
check = []
# print(level_guid.index('-1'))
# exit(0)

# level = ['5th Storey', '7th Storey', '8th Storey', '9th Storey', '10th Storey', '11th Storey', '12th Storey']
# level = ['1st Storey', '2nd Storey', '3rd Storey', '4th Storey', '5th Storey', '6th Storey', '7th Storey',
#          'Top Highest Level', 'Platform Level', '8th Storey', '9th Storey', '10th Storey', '11th Storey', '12th Storey',
#          'Roof Level']
# l = [694]
# level = ['2nd Storey']

level = ['2ND STOREY']

for key in level:
    if key in check:
        continue
    if key in level_guid:
        index_ = level_guid.index(key)
        df_process_final = df_process_final.append(df_process_group_level.get_group(key))
        df_process_final = df_process_final.append(df_process_group_level.get_group(level_name[index_]))
        df_process_final = df_process_final.append(df_process_group_base_constraint.get_group(int(level_guid[index_])))
        check.extend((key, level_name[index_]))
        print("SHAPE OF DATA FOR EACH LEVEL GUID {}  {}".format(key, df_process_final.shape))
        df_process_2 = Plot(df_process_final, level_name[index_], attributes)
        # print(check)
        df_process_final = pd.DataFrame()

    if key in level_name:
        index_ = level_name.index(key)
        for attr in attributes:
            if attr == 'Rooms':
                df_process_final = df_process_final.append(df_process_group_level.get_group(key))
            if attr == 'Rooms' or 'Floors' or 'Doors':
                df_process_final = df_process_final.append(df_process_group_level.get_group(int(level_guid[index_])))
            if attr == 'Walls':
                df_process_final = df_process_final.append(
                    df_process_group_base_constraint.get_group(int(level_guid[index_])))
        check.extend((key, level_guid[index_]))
        print("SHAPE OF DATA FOR LEVEL {}  {}".format(key, df_process_final.shape))

        df_process_final = df_process_final.groupby(('Geometry', 'Type')).get_group('Mesh')
        # Plot(df_process_final, level_name[index_], df_process_2, attributes)
        df_process_2 = Plot(df_process_final, level_name[index_], df_process_2, attributes)
        # print(check)
        df_process_final = pd.DataFrame()

print(df_process_2.head())

import os

path_export = r'C:\Users\sutd\Desktop\SMU_NEW\Processed.xlsx'
if os.path.exists(path_export):
    os.remove(path_export)
    print("File removal successfull")
else:
    print("The file does not exist")
df_process_2.to_excel(path_export, header=True, index=True)
# df_geo = pd.DataFrame()
# df_geo['geometry'] = df_process_2[('Geometry', 'Shape')]
# # print(df_process_2.head())
# gdf = gpd.GeoDataFrame(df_geo, geometry="geometry")
# gdf.plot(color="white", edgecolor='black')
# plt.show()

exit(0)
# level = "1st Storey"
# df_process = df_process_group.get_group(level)
# print(df_process.shape)
# Plot(df_process, level)
