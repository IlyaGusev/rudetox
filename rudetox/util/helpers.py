def get_first_elements(elements):
    used_elements = set()
    first_elements = []
    for elem in elements:
        if elem not in used_elements:
            first_elements.append(elem)
            used_elements.add(elem)
    return first_elements
