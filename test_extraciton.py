def parse_input_string(input_str):
    # Split the string by $$$ delimiters
    sections = [section.strip() for section in input_str.split('$$$$') if section.strip()]
    
    result_dict = {}
    i = 0
    while i < len(sections):
        key = sections[i]
        if key == 'Y':
            # For Y keys, the next section is the value
            if i + 1 < len(sections):
                value = sections[i+1]
                result_dict[key] = value
                i += 2
            else:
                i += 1
        elif key == 'T':
            # For T keys, the next sections form a list value
            values = []
            i += 1
            while i < len(sections) and sections[i] not in ['Y', 'T']:
                # Try to convert to int if possible
                try:
                    values.append(int(sections[i]))
                except ValueError:
                    values.append(sections[i])
                i += 1
            result_dict[key] = values
        else:
            i += 1
    
    return result_dict

# Example usage
input_string = "Y$$$$$$$$ lofi music$$$$$$$$$$$$ T$$$$$$$$'bedtime reminder'$$$$$$$$81000$$$$$$$$"
parsed_dict = parse_input_string(input_string)
print(parsed_dict)