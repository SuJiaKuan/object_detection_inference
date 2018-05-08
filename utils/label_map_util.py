"""Label map utility functions."""

def create_category_index(labels_path):
    """Create the index of label categories"""
    with open(labels_path) as f:
        lines = f.readlines()
    category_index = dict()
    for line in lines:
        # Format: "id label"
        splits = line.strip().split(' ')
        category_index[int(splits[0])] = splits[1]
    return category_index
