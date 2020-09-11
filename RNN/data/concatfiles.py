from pathlib import Path

for path in Path('linux-master').rglob('*.c'):
    print(path)
    fin = open(path, "r")
    data2 = fin.read()
    fin.close()
    fout = open("data.c", "a")
    fout.write(data2)
    fout.close()
