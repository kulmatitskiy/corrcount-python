# Misc

def remove_indices(l, ii):
    ii = set(ii)
    return [e for i, e in enumerate(l) if i not in ii]