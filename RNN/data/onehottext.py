
# this is probably hella slow but it's fast enough for me


def one_hot_text_data(path, size=100000):
    data_file = open(path, "r")
    data_str = data_file.read()
    if size == -1:
        data_subset = data_str[:]
    else:
        data_subset = data_str[:size]
    print(len(data_subset))

    chars_used = []
    for char in data_subset:
        if not chars_used.__contains__(char):
            chars_used.append(char)

    one_hots = []
    for char in data_subset:
        one_hots.append([])
        for i in range(len(chars_used)):
            one_hots[-1].append(0)
        index = chars_used.index(char)
        one_hots[-1][index] = 1

    if __name__ == '__main__':
        print("There were {} unique chars".format(chars_used.__len__()))
    return one_hots


if __name__ == '__main__':
    one_hot_data = one_hot_text_data(50)
    print(one_hot_data)
