
[[Natural Language Processing|Natural language generation]] involves generating an output token from a sequence of input tokens. The 'tokens' are words in the case of natural language, but what if one generated 3D structures where the tokens are voxels. 

## Current status of the project:

Download some .ply file, load into custom `OctreeData` class, which (using Open3d) converts into a list of voxels and an octree. Using the Open3d octree class, one can efficiently query a 3d coordinate and get the voxel colour. 

The `OctreeData` class calculates the set of unique colours used to represent the 3D structure (TODO: need some clustering to discretize any blending colours if this occurs). Then `OctreeData.coord_to_colour_id(x, y, z)` maps a given coordinate to its voxel's colour's index in the set of colours, from 1 to N. The index of 0 is reserved for empty space. This is out set of tokens: "0" to "N". 

With natural language generation, we predict following tokens from an input sequence which is the earlier part of a sentence, e.g. {'Andrew', 'is', 'a'} -> {'legend'}. With our voxel colour tokens, we might want to predict the output token for a given coordinate, from input tokens which are the set of adjacent tokens to this coordinate (or within some distance of the coordinate). Generating the next token would be something like: {"0", "0", "2", "11", "6", "3"} -> {"2"}. 

More specifically, to generate a whole 3D structure one would loop over all coordinates we're interested in, so out input tokens would be the previously generated 'adjacent' tokens. As an example, consider this order of 3 predictions in a simple 2D case with input tokens 'x' and output tokens 'o', otherwise 'n':

	o n n -> x o n -> n x o
	x n n    n x n    n n x


