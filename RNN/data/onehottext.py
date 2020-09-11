

def one_hot_text_data(size=100000):
    data_file = open("data.c", "r")
    data_str = data_file.read()
    data_subset = data_str[:size]

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
