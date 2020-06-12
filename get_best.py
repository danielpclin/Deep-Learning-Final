import numpy as np


def find_best(dataset=1, start=1, end=10, nums=5):
    data = []
    data_val = []
    for i in range(start, end+1):
        try:
            with open(f"data0{dataset}_{i}.txt", "r") as file:
                for _ in range(13):
                    file.readline()
                data.append(i)
                data_val.append(float(file.readline()))
        except FileNotFoundError:
            pass
    a = np.argsort(data_val)
    for i in range(min(nums, len(data_val))):
        # print(data[a[i]])
        print(data[a[-i-1]])
        print(data_val[a[-i-1]])
    for i in range(min(nums, len(data_val))):
        print(data[a[-i-1]], end=", ")
    print()


def main():
    dataset = int(input("dataset: "))
    start = int(input("start: "))
    end = int(input("end: "))
    num = int(input("num: "))
    find_best(dataset=dataset, start=start, end=end, nums=num)


if __name__ == "__main__":
    main()
