from data.onehottext import one_hot_str


def one_hot_genome(path):
    data_file = open(path, "rb")
    data_bytes = data_file.read(1000000)
    data_str = data_bytes.decode("utf-8").upper()
    filtered_str = ""
    for i in range(0, len(data_str)):
        if data_str[i] == 'A' or data_str[i] == 'G' or data_str[i] == 'C' or data_str[i] == 'T':
            filtered_str += data_str[i]
    return one_hot_str(filtered_str)
