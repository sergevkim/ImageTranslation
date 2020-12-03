def main():
    n = int(input())
    k = sorted(list(map(int, input().split())), reverse=True)
    print(k)

    for i, ki in enumerate(k):
        if ki < i + 1:
            print(i)
            return

    print()


main()

