# 2019-12-19 Utils for Tree Tensor Network


def gen_basis_location_list(n):
	layer_list = []
	layer_0_list = [[j] for j in range(2 ** n)]
	layer_list.append(layer_0_list)

	for i in range(1, n + 1):
		layer_list_temp = [ [layer_list[i - 1][j][-1], layer_list[i - 1][j + 1][-1]] for j in range(0,len(layer_list[i - 1]),2)]
		layer_list.append(layer_list_temp)

	return layer_list[1:]


def gen_modifiable_location_list(basis_location_list, n):
	locations_to_be_modified = []
	for i in range(0,n-1):
		locations_to_be_modified_temp = [ [basis_location_list[i][j][-1], basis_location_list[i][j + 1][0]] for j in range(0, len(basis_location_list[i]) - 1, 1)]
		locations_to_be_modified.append(locations_to_be_modified_temp)

	return locations_to_be_modified



def generate_tree_config(n):
	fix_layer_list = gen_basis_location_list(n)
	fix_layer_list_out = []
	for item in fix_layer_list:
		for loc in item:
			fix_layer_list_out.append(loc)

	locations_to_be_modified = gen_modifiable_location_list(fix_layer_list, n)
	locations_to_be_modified_out = []
	for item in locations_to_be_modified:
		for loc in item:
			locations_to_be_modified_out.append(loc)

	return fix_layer_list_out, locations_to_be_modified_out